"""Run images through frozen DINOv2 and save one embedding per image.

No training happens here: DINO already learned general-purpose features from
~142M unlabeled images, so we just use it as a feature extractor.

Datasets:
    pets     Oxford-IIIT Pets, 37 cat/dog breeds (downloads automatically)
    cifar10  CIFAR-10, 10 everyday classes (downloads automatically)
    folder   your own images, arranged as <data>/<class_name>/<image>.jpg

Usage:
    python embed.py --dataset pets
    python embed.py --dataset folder --data my_images
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from dino_utils import dino_features, load_dino, make_transform


def get_splits(name, root, tf):
    """Return (train_ds, test_ds, class_names). test_ds is None for 'folder'."""
    if name == "pets":
        tr = datasets.OxfordIIITPet(root, split="trainval", transform=tf, download=True)
        te = datasets.OxfordIIITPet(root, split="test", transform=tf, download=True)
        return tr, te, tr.classes
    if name == "cifar10":
        tr = datasets.CIFAR10(root, train=True, transform=tf, download=True)
        te = datasets.CIFAR10(root, train=False, transform=tf, download=True)
        return tr, te, tr.classes
    ds = datasets.ImageFolder(root, transform=tf)
    return ds, None, ds.classes


def subsample(ds, n, seed=0):
    if n is None or n >= len(ds):
        return ds
    idx = np.random.default_rng(seed).choice(len(ds), n, replace=False)
    return torch.utils.data.Subset(ds, idx.tolist())


@torch.no_grad()
def embed(model, device, ds, batch_size, desc):
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2)
    feats, labels = [], []
    for imgs, ys in tqdm(loader, desc=desc):
        cls, _ = dino_features(model, imgs.to(device))
        feats.append(cls.cpu().numpy())
        labels.append(np.asarray(ys))
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pets", choices=["pets", "cifar10", "folder"])
    ap.add_argument("--data", default="data", type=Path, help="Download dir, or your image folder")
    ap.add_argument("--model", default="small", choices=["small", "base", "large"])
    ap.add_argument("--img-size", default=224, type=int)
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--max-train", default=None, type=int,
                    help="Use a random subset to go faster (e.g. 5000 for CIFAR-10 on CPU)")
    ap.add_argument("--max-test", default=None, type=int)
    ap.add_argument("--out", default="results/embeddings.npz", type=Path)
    args = ap.parse_args()

    model, device = load_dino(args.model)
    print(f"Loaded DINOv2-{args.model} on {device}")

    train_ds, test_ds, classes = get_splits(args.dataset, args.data, make_transform(args.img_size))
    train_ds = subsample(train_ds, args.max_train)
    out = {"classes": np.array(classes), "model": args.model,
           "img_size": args.img_size, "dataset": args.dataset}

    out["X_train"], out["y_train"] = embed(model, device, train_ds, args.batch_size, "Embedding train")
    if test_ds is not None:
        test_ds = subsample(test_ds, args.max_test)
        out["X_test"], out["y_test"] = embed(model, device, test_ds, args.batch_size, "Embedding test")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"Saved embeddings for {len(classes)} classes to {args.out}")


if __name__ == "__main__":
    main()
