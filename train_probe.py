"""Train tiny classifiers on top of frozen DINOv2 embeddings and report results.

Two standard ways to evaluate DINO features (both from the DINO papers):
  * k-NN:          label an image by its nearest neighbours in embedding space
  * linear probe:  a single logistic-regression layer on the embedding

Usage:
    python train_probe.py --embeddings results/embeddings.npz
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="results/embeddings.npz", type=Path)
    ap.add_argument("--test-size", default=0.2, type=float)
    ap.add_argument("--k", default=10, type=int)
    ap.add_argument("--out-dir", default="results", type=Path)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()

    d = np.load(args.embeddings, allow_pickle=False)
    classes = [str(c) for c in d["classes"]]
    if "X_test" in d:  # official split from the dataset
        X_tr, y_tr, X_te, y_te = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    else:              # own image folder: make a random split
        X_tr, X_te, y_tr, y_te = train_test_split(
            d["X_train"], d["y_train"], test_size=args.test_size,
            stratify=d["y_train"], random_state=args.seed)
    print(f"Train {len(y_tr)} / test {len(y_te)} images, {len(classes)} classes")

    # k-NN with cosine similarity (L2-normalize, then Euclidean == cosine ranking)
    knn = make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=args.k, weights="distance"))
    knn.fit(X_tr, y_tr)
    knn_pred = knn.predict(X_te)

    # Linear probe
    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
    probe.fit(X_tr, y_tr)
    probe_pred = probe.predict(X_te)

    metrics = {}
    for name, pred in [("knn", knn_pred), ("linear_probe", probe_pred)]:
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        }
        print(f"\n== {name} ==  acc {metrics[name]['accuracy']:.3f}  macro-F1 {metrics[name]['macro_f1']:.3f}")
    print("\nLinear probe per-class report:")
    print(classification_report(y_te, probe_pred, target_names=classes, digits=3, zero_division=0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"model": probe, "classes": classes,
                 "dino_model": str(d["model"]), "img_size": int(d["img_size"])},
                args.out_dir / "probe.joblib")

    # Confusion matrix (numbers in cells only when there are few classes)
    n = len(classes)
    side = min(0.45 * n + 4, 18)
    fig, ax = plt.subplots(figsize=(side, side))
    ConfusionMatrixDisplay.from_predictions(
        y_te, probe_pred, display_labels=classes, normalize="true", xticks_rotation=90,
        cmap="Blues", values_format=".2f" if n <= 12 else None,
        include_values=n <= 12, ax=ax, colorbar=False)
    ax.set_title("DINOv2 linear probe — confusion matrix (row-normalized)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # t-SNE of test embeddings: shows DINO already clusters classes without labels
    X, y = X_te[:3000], y_te[:3000]
    perplexity = min(30, max(5, len(X) // 4))
    emb2d = TSNE(n_components=2, perplexity=perplexity, init="pca",
                 random_state=args.seed).fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab20" if len(classes) > 10 else "tab10")
    for i, c in enumerate(classes):
        m = y == i
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=6, alpha=0.7, label=c,
                   color=cmap(i % cmap.N), marker="o" if i < cmap.N else "^")
    if len(classes) <= 20:
        ax.legend(markerscale=2, fontsize=7, loc="best")
    ax.set_title("t-SNE of DINOv2 embeddings (no fine-tuning)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out_dir / "tsne.png", dpi=150)
    plt.close(fig)

    print(f"Saved metrics.json, probe.joblib, confusion_matrix.png, tsne.png to {args.out_dir}/")


if __name__ == "__main__":
    main()
