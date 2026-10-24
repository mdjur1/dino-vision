"""Image search with DINOv2: find the images most similar to a query photo.

Embeds a gallery of images once (cached to disk), then ranks them by cosine
similarity to the query's DINOv2 embedding. No training involved.

Usage:
    python find_similar.py data/oxford-iiit-pet/images/pug_10.jpg
    python find_similar.py my_dog.jpg --top 8
    python find_similar.py photo.jpg --gallery my_photos --max-gallery 500
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dino_utils import dino_features, embed_paths, load_dino, load_image, make_transform

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder, max_n, seed):
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No images found in {folder}")
    if max_n and len(paths) > max_n:
        idx = np.random.default_rng(seed).choice(len(paths), max_n, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    return paths


def load_or_build_gallery(model, device, tf, args):
    paths = list_images(args.gallery, args.max_gallery, args.seed)
    cache = args.cache or Path(f"results/gallery_{args.model}_{len(paths)}.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if [str(p) for p in paths] == list(d["requested"]):
            print(f"Using cached gallery embeddings from {cache}")
            return d["features"], [Path(p) for p in d["paths"]]
    print(f"Embedding {len(paths)} gallery images (only happens once)...")
    feats, kept = embed_paths(model, device, paths, tf, batch_size=args.batch_size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, features=feats, paths=np.array([str(p) for p in kept]),
             requested=np.array([str(p) for p in paths]))
    return feats, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Query image (a wildcard like beagle_1*.jpg uses the first match)")
    ap.add_argument("--gallery", default="data/oxford-iiit-pet/images", type=Path)
    ap.add_argument("--max-gallery", default=2000, type=int, help="0 = use every image")
    ap.add_argument("--top", default=5, type=int)
    ap.add_argument("--model", default="small", choices=["small", "base", "large"])
    ap.add_argument("--img-size", default=224, type=int)
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--cache", default=None, type=Path)
    ap.add_argument("--out", default="results/similar.png", type=Path)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()

    matches = sorted(glob.glob(args.query))
    if not matches:
        raise SystemExit(f"Query image not found: {args.query}")
    query = Path(matches[0])

    model, device = load_dino(args.model)
    tf = make_transform(args.img_size)
    feats, paths = load_or_build_gallery(model, device, tf, args)

    q, _ = dino_features(model, tf(load_image(query)).unsqueeze(0).to(device))
    q = q.cpu().numpy()[0]

    # Cosine similarity = dot product of L2-normalized vectors
    g = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    sims = g @ (q / np.linalg.norm(q))
    order = [i for i in np.argsort(-sims) if paths[i].resolve() != query.resolve()][: args.top]

    print(f"\nMost similar to {query.name}:")
    for rank, i in enumerate(order, 1):
        print(f"  {rank}. {paths[i].name}  (similarity {sims[i]:.3f})")

    fig, axes = plt.subplots(1, args.top + 1, figsize=(2.6 * (args.top + 1), 3.2))
    axes[0].imshow(load_image(query))
    axes[0].set_title(f"QUERY\n{query.name}", fontsize=8, fontweight="bold")
    for ax, i in zip(axes[1:], order):
        ax.imshow(load_image(paths[i]))
        ax.set_title(f"{paths[i].name}\nsim {sims[i]:.2f}", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("DINOv2 image search (cosine similarity of CLS embeddings)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
