import argparse
import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_service.model import ARCHITECTURE, MEAN, STD, MNISTNet


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/data"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.workers < 0 or args.learning_rate <= 0:
        parser.error("Invalid training parameters")
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    train = datasets.MNIST(output / "dataset", train=True, download=True, transform=transform)
    test = datasets.MNIST(output / "dataset", train=False, download=True, transform=transform)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        generator=torch.Generator().manual_seed(args.seed))
    test_loader = DataLoader(test, batch_size=256, num_workers=args.workers)
    model = MNISTNet().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = correct = count = 0
        epoch_start = time.perf_counter()
        for inputs, labels in loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (predictions.argmax(1) == labels).sum().item()
            count += len(labels)
        model.eval()
        test_loss = test_correct = evaluated = 0
        with torch.inference_mode():
            for inputs, labels in test_loader:
                predictions = model(inputs)
                test_loss += criterion(predictions, labels).item() * len(labels)
                test_correct += (predictions.argmax(1) == labels).sum().item()
                evaluated += len(labels)
        result = dict(epoch=epoch + 1, train_loss=loss_sum / count, train_accuracy=correct / count,
                      test_loss=test_loss / evaluated, test_accuracy=test_correct / evaluated,
                      duration_seconds=time.perf_counter() - epoch_start)
        epochs.append(result)
        print(json.dumps(result), flush=True)
    temporary = output / "model.pt.tmp"
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, output / "model.pt")
    examples = output / "examples"
    examples.mkdir(exist_ok=True)
    index = []
    seen = set()
    for i, label in enumerate(test.targets.tolist()):
        if label in seen:
            continue
        seen.add(label)
        identifier = str(i)
        Image = transforms.ToPILImage()
        Image(test.data[i]).save(examples / f"{identifier}.png")
        index.append(dict(id=identifier, label=label, split="test"))
        if len(seen) == 10:
            break
    atomic_json(examples / "index.json", index)
    parameters = {k: v for k, v in vars(args).items() if k != "output_dir"}
    metrics = dict(dataset="MNIST", train_examples=len(train), evaluated_examples=len(test),
                   parameters=parameters, epochs=epochs, test_accuracy=epochs[-1]["test_accuracy"],
                   test_loss=epochs[-1]["test_loss"], started_at=started,
                   ended_at=datetime.now(timezone.utc).isoformat(), duration_seconds=time.perf_counter() - start,
                   versions=dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__))
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "metadata.json", dict(architecture=ARCHITECTURE,
                checkpoint_sha256=hashlib.sha256((output / "model.pt").read_bytes()).hexdigest(),
                trained_at=metrics["ended_at"], parameters=parameters,
                input_convention=dict(shape=[1, 28, 28], foreground="light on dark", mean=MEAN, std=STD)))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
