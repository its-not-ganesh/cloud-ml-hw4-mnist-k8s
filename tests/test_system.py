import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageOps
from mnist_service.app import create_app
from mnist_service.model import ARCHITECTURE, MNISTNet, normalize
from mnist_service.preprocess import uploaded_pixels


@pytest.fixture
def artifacts(tmp_path):
    torch.manual_seed(42)
    model = MNISTNet().eval()
    torch.save(model.state_dict(), tmp_path / "model.pt")
    checksum = hashlib.sha256((tmp_path / "model.pt").read_bytes()).hexdigest()
    (tmp_path / "metadata.json").write_text(json.dumps(dict(architecture=ARCHITECTURE, checkpoint_sha256=checksum)))
    (tmp_path / "metrics.json").write_text('{}')
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples/index.json").write_text('[{"id":"0","label":7,"split":"test"}]')
    Image.new("L", (28, 28), 100).save(tmp_path / "examples/0.png")
    return tmp_path, model


def test_model_roundtrip(artifacts):
    path, original = artifacts
    inputs = normalize(np.ones((28, 28)))
    loaded = MNISTNet().eval()
    loaded.load_state_dict(torch.load(path / "model.pt", weights_only=True, map_location="cpu"))
    assert loaded(inputs).shape == (1, 10)
    assert torch.equal(original(inputs), loaded(inputs))


def image_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_preprocess():
    image = Image.new("L", (80, 80))
    ImageDraw.Draw(image).line((30, 10, 40, 70), fill=255, width=8)
    pixels = uploaded_pixels(image_bytes(image))
    inverted = uploaded_pixels(image_bytes(ImageOps.invert(image)))
    assert pixels.shape == (28, 28)
    assert np.array_equal(pixels, inverted)
    assert np.array_equal(pixels, uploaded_pixels(image_bytes(image)))
    assert pixels.min() == 0 and pixels.max() <= 1
    assert np.allclose(normalize(np.full((28, 28), 0.1307)).numpy(), 0, atol=1e-6)


def test_missing_checkpoint(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/healthz').status_code == 200
        assert client.get('/readyz').status_code == 503
        assert client.post('/api/predict/drawn', json={'pixels': [0] * 784}).status_code == 503


def test_corrupt_checkpoint(artifacts):
    path, _ = artifacts
    (path / 'model.pt').write_bytes(b'corrupt')
    with TestClient(create_app(path)) as client:
        assert client.get('/readyz').status_code == 503


def test_real_forward_pass_and_examples(artifacts):
    path, model = artifacts
    with TestClient(create_app(path)) as client:
        assert client.get('/').status_code == 200
        assert client.get('/readyz').status_code == 200
        response = client.post('/api/predict/drawn', json={'pixels': [0.5] * 784})
        result = response.json()
        with torch.inference_mode():
            expected = model(normalize([0.5] * 784)).softmax(1)[0].tolist()
        assert result['probabilities'] == pytest.approx(expected)
        assert sum(result['probabilities']) == pytest.approx(1)
        assert result['inference_ms'] > 0
        assert client.get('/api/examples').json()[0]['split'] == 'test'
        assert client.post('/api/predict/example/0').json()['actual_label'] == 7
        assert client.post('/api/predict/example/999').status_code == 404
        assert client.get('/api/examples/0/image').headers['content-type'] == 'image/png'
        assert client.get('/api/metadata').json()['model']['architecture'] == ARCHITECTURE
        uploaded = client.post('/api/predict', files={'file': ('digit.png', (path / 'examples/0.png').read_bytes(), 'image/png')})
        # A uniform image is deliberately rejected as blank after border inversion.
        assert uploaded.status_code == 422


@pytest.mark.parametrize('pixels', [[0] * 783, [0] * 785, [-1] * 784, [1.01] * 784, ['nan'] * 784, ['inf'] * 784])
def test_invalid_drawings(artifacts, pixels):
    with TestClient(create_app(artifacts[0])) as client:
        assert client.post('/api/predict/drawn', json={'pixels': pixels}).status_code == 422


def test_upload_validation_and_inference(artifacts):
    with TestClient(create_app(artifacts[0])) as client:
        assert client.post('/api/predict', files={'file': ('x.txt', b'bad', 'text/plain')}).status_code == 415
        assert client.post('/api/predict', files={'file': ('x.png', b'bad', 'image/png')}).status_code == 422
        assert client.post('/api/predict', files={'file': ('x.png', b'x' * 2_100_000, 'image/png')}).status_code == 413
        image = Image.new('L', (28, 28))
        ImageDraw.Draw(image).line((8, 4, 18, 24), fill=255, width=3)
        assert client.post('/api/predict', files={'file': ('x.png', image_bytes(image), 'image/png')}).status_code == 200


def test_manifest_contracts():
    resources = {p.name: yaml.safe_load(p.read_text()) for p in Path('k8s/gke').glob('*.yaml')}
    assert len(resources) == 5
    deployment = resources['inference-deployment.yaml']['spec']
    service = resources['inference-service.yaml']['spec']
    assert deployment['replicas'] == 1
    assert deployment['template']['metadata']['labels'] == service['selector']
    assert service['type'] == 'LoadBalancer'
    assert resources['pvc.yaml']['spec']['storageClassName'] == 'standard-rwo'
    for file in ['training-job.yaml', 'inference-deployment.yaml']:
        spec = resources[file]['spec']['template']['spec']
        assert spec['securityContext']['runAsNonRoot']
        assert not any('gpu' in k for k in spec['containers'][0]['resources']['limits'])
