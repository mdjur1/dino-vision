"""The classic DINO visualization: PCA on patch features -> RGB.

DINOv2 gives one feature vector per 14x14 patch. Running PCA on those vectors
across a few images and mapping the top 3 components to RGB makes matching
object parts light up in matching colors, with no labels or training at all.

  1. First PCA component separates foreground (the object) from background.
  2. A second PCA on foreground patches only -> 3 components -> RGB.

Works best with several photos of the same kind of thing (e.g. 4 dogs).

Usage:
    python patch_pca.py data/oxford-iiit-pet/images/beagle_1*.jpg --max-images 4
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from dino_utils import PATCH_SIZE, dino_features, load_dino, load_image, make_transform


def minmax(x):
    lo, hi = x.min(axis=0), x.max(axis=0)
    return (x - lo) / (hi - lo + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--model", default="small", choices=["small", "base", "large"])
    ap.add_argument("--img-size", default=448, type=int, help="Bigger = finer patch grid")
    ap.add_argument("--max-images", default=4, type=int)
    ap.add_argument("--threshold", default=0.5, type=float,
                    help="Foreground cutoff on the (0-1 scaled) first PCA component")
    ap.add_argument("--out", default="results/patch_pca.png", type=Path)
    args = ap.parse_args()

    # Expand wildcards ourselves (Windows cmd doesn't do it like Mac/Linux shells)
    paths = []
    for p in args.images:
        matches = sorted(glob.glob(str(p)))
        if not matches:
            raise SystemExit(f"No images found matching: {p}")
        paths.extend(Path(m) for m in matches)
    paths = paths[: args.max_images]
    model, device = load_dino(args.model)
    tf = make_transform(args.img_size)
    grid = args.img_size // PATCH_SIZE

    raw = [load_image(p) for p in paths]
    batch = torch.stack([tf(im) for im in raw]).to(device)
    _, patches = dino_features(model, batch)          # [B, grid*grid, D]
    B, N, D = patches.shape
    feats = patches.reshape(B * N, D).cpu().numpy()

    # Step 1: foreground vs background from the first principal component
    pc1 = minmax(PCA(n_components=1).fit_transform(feats))[:, 0]
    # PCA sign is arbitrary: flip so the image border (usually background) is "low"
    pc1_maps = pc1.reshape(B, grid, grid)
    border = np.concatenate([pc1_maps[:, 0], pc1_maps[:, -1], pc1_maps[:, :, 0], pc1_maps[:, :, -1]], axis=1)
    if border.mean() > pc1.mean():
        pc1 = 1 - pc1
    fg = pc1 > args.threshold

    # Step 2: 3-component PCA on foreground patches -> RGB
    rgb = np.zeros((B * N, 3), dtype=np.float32)
    if fg.sum() >= 3:
        rgb[fg] = minmax(PCA(n_components=3).fit_transform(feats[fg]))
    rgb = rgb.reshape(B, grid, grid, 3)

    fig, axes = plt.subplots(2, B, figsize=(3.2 * B, 6.6), squeeze=False)
    for i, im in enumerate(raw):
        shown = im.resize((args.img_size, args.img_size))
        axes[0, i].imshow(shown)
        axes[1, i].imshow(rgb[i], interpolation="nearest")
        axes[0, i].set_title(paths[i].name, fontsize=8)
        for a in axes[:, i]:
            a.axis("off")
    fig.suptitle(f"DINOv2-{args.model} patch features → PCA → RGB (no labels, no training)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
