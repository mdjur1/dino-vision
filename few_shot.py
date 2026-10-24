"""How many labels does DINO actually need?

Trains the linear probe and k-NN with only N labeled images per class
(N = 1, 2, 5, 10, 20, 50, all) and plots test accuracy against N.
Small N values are repeated over several random draws and averaged.

Uses the embeddings already saved by embed.py, so it runs in a minute or two.

Usage:
    python few_shot.py
    python few_shot.py --shots 1 5 10 --trials 10
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


def load_split(path, test_size=0.2, seed=0):
    d = np.load(path, allow_pickle=False)
    classes = [str(c) for c in d["classes"]]
    if "X_test" in d:
        return d["X_train"], d["y_train"], d["X_test"], d["y_test"], classes
    X_tr, X_te, y_tr, y_te = train_test_split(
        d["X_train"], d["y_train"], test_size=test_size,
        stratify=d["y_train"], random_state=seed)
    return X_tr, y_tr, X_te, y_te, classes


def sample_per_class(y, n, rng):
    """Pick n random training indices from every class."""
    idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        idx.extend(rng.choice(members, min(n, len(members)), replace=False))
    return np.array(idx)


def fit_and_score(X_tr, y_tr, X_te, y_te, k):
    knn = make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    knn.fit(X_tr, y_tr)
    probe.fit(X_tr, y_tr)
    return knn.score(X_te, y_te), probe.score(X_te, y_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="results/embeddings.npz", type=Path)
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50])
    ap.add_argument("--trials", default=5, type=int, help="Random draws per shot count")
    ap.add_argument("--k", default=10, type=int, help="Max neighbours for k-NN")
    ap.add_argument("--out-dir", default="results", type=Path)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()
    # sklearn warns when there are fewer than 2 samples per class; that's the point here
    warnings.filterwarnings("ignore", message="The number of unique classes")

    X_tr, y_tr, X_te, y_te, classes = load_split(args.embeddings, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    full_per_class = int(np.bincount(y_tr).min())
    shots = sorted(s for s in set(args.shots) if s < full_per_class)

    rows = []
    for n in shots:
        knn_acc, probe_acc = [], []
        for _ in range(args.trials):
            idx = sample_per_class(y_tr, n, rng)
            a, b = fit_and_score(X_tr[idx], y_tr[idx], X_te, y_te, k=min(args.k, n))
            knn_acc.append(a)
            probe_acc.append(b)
        rows.append({"shots": n,
                     "knn_mean": float(np.mean(knn_acc)), "knn_std": float(np.std(knn_acc)),
                     "probe_mean": float(np.mean(probe_acc)), "probe_std": float(np.std(probe_acc))})
        print(f"{n:>4} per class: k-NN {rows[-1]['knn_mean']:.3f}   linear probe {rows[-1]['probe_mean']:.3f}")

    # Full training set, one run
    a, b = fit_and_score(X_tr, y_tr, X_te, y_te, k=args.k)
    rows.append({"shots": "all", "knn_mean": float(a), "knn_std": 0.0,
                 "probe_mean": float(b), "probe_std": 0.0})
    print(f" all ({len(y_tr)} imgs): k-NN {a:.3f}   linear probe {b:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "few_shot.json").write_text(json.dumps(rows, indent=2))

    # Plot: x positions are the shot counts, plus one slot for "all"
    xs = [r["shots"] for r in rows[:-1]] + [round(len(y_tr) / len(classes))]
    labels = [str(r["shots"]) for r in rows[:-1]] + ["all"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, name, marker in [("probe", "Linear probe", "o"), ("knn", "k-NN", "s")]:
        m = np.array([r[f"{key}_mean"] for r in rows]) * 100
        s = np.array([r[f"{key}_std"] for r in rows]) * 100
        ax.plot(xs, m, marker=marker, label=name)
        ax.fill_between(xs, m - s, m + s, alpha=0.2)
    ax.axhline(100 / len(classes), color="gray", ls="--", lw=1, label="Random guess")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.minorticks_off()
    ax.set_xlabel("Labeled training images per class")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Frozen DINOv2: accuracy vs. number of labels")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out_dir / "few_shot.png", dpi=150)
    print(f"Saved few_shot.json and few_shot.png to {args.out_dir}/")


if __name__ == "__main__":
    main()
