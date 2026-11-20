"""Render an immutable, evaluated image pair without editing source manifests."""
import argparse
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("--sha", required=True)
parser.add_argument("--output", type=Path, default=Path("rendered"))
args = parser.parse_args()
if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
    parser.error("Expected full 40-character commit SHA")
args.output.mkdir(parents=True, exist_ok=True)
for path in Path("k8s/gke").glob("*.yaml"):
    text = path.read_text().replace(":main", f":sha-{args.sha}")
    (args.output / path.name).write_text(text, encoding="utf-8")
