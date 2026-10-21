"""Classify new images with the trained probe.

Usage:
    python predict.py path/to/photo1.jpg path/to/photo2.jpg
"""

import argparse
from pathlib import Path

import joblib
import torch

from dino_utils import dino_features, load_dino, load_image, make_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--probe", default="results/probe.joblib", type=Path)
    args = ap.parse_args()

    bundle = joblib.load(args.probe)
    probe, classes = bundle["model"], bundle["classes"]
    model, device = load_dino(bundle["dino_model"])
    tf = make_transform(bundle["img_size"])

    batch = torch.stack([tf(load_image(p)) for p in args.images]).to(device)
    cls, _ = dino_features(model, batch)
    probs = probe.predict_proba(cls.cpu().numpy())

    for path, p in zip(args.images, probs):
        top = p.argsort()[::-1][:3]
        summary = ", ".join(f"{classes[i]} {p[i]:.0%}" for i in top)
        print(f"{path.name}: {summary}")


if __name__ == "__main__":
    main()
