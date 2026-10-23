"""Live DINOv2 demo with OpenCV: webcam (or a video file) in, DINO's view out.

Two modes (press M to switch):
  classify  top-3 predictions from the linear probe trained by train_probe.py
  pca       live patch-feature PCA: DINO's patch features mapped to colors,
            shown next to the camera feed. Colors are fitted on one frame and
            then kept fixed, so the same object part keeps the same color.

Keys:  M = switch mode   R = refit PCA colors on current frame
       S = save screenshot to results/   Q or Esc = quit

Usage:
    python webcam_demo.py                      # default webcam
    python webcam_demo.py --camera 1           # another camera
    python webcam_demo.py --video clip.mp4     # a video file instead
    python webcam_demo.py --mode pca
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from dino_utils import PATCH_SIZE, dino_features, load_dino, make_transform


# ---------------------------------------------------------------- helpers

def center_square(frame):
    """Crop the middle square of a frame (DINO's preprocessing also center-crops)."""
    h, w = frame.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return frame[y:y + s, x:x + s]


def put_text(img, text, org, scale=0.6, color=(255, 255, 255)):
    """Text with a dark outline so it's readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class LivePCA:
    """Two-stage patch PCA (same idea as patch_pca.py) with colors frozen after fitting."""

    def __init__(self, grid, threshold=0.5):
        self.grid, self.threshold = grid, threshold
        self.fitted = False

    def fit(self, feats):
        self.pca1 = PCA(n_components=1).fit(feats)
        s = self.pca1.transform(feats)[:, 0]
        # PCA sign is arbitrary: make the border (usually background) low
        m = s.reshape(self.grid, self.grid)
        border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
        self.sign = -1.0 if border.mean() > s.mean() else 1.0
        s = self.sign * s
        self.lo1, self.hi1 = s.min(), s.max()
        fg = self._norm1(s) > self.threshold
        if fg.sum() < 4:  # nothing stands out: use every patch
            fg[:] = True
        self.pca3 = PCA(n_components=3).fit(feats[fg])
        c = self.pca3.transform(feats[fg])
        self.lo3, self.hi3 = c.min(axis=0), c.max(axis=0)
        self.fitted = True

    def _norm1(self, s):
        return np.clip((s - self.lo1) / (self.hi1 - self.lo1 + 1e-8), 0, 1)

    def apply(self, feats):
        """Return an RGB uint8 image of shape (grid, grid, 3)."""
        if not self.fitted:
            self.fit(feats)
        fg = self._norm1(self.sign * self.pca1.transform(feats)[:, 0]) > self.threshold
        rgb = np.clip((self.pca3.transform(feats) - self.lo3) / (self.hi3 - self.lo3 + 1e-8), 0, 1)
        rgb[~fg] = 0
        return (rgb.reshape(self.grid, self.grid, 3) * 255).astype(np.uint8)


def load_probe(path):
    if not Path(path).exists():
        return None
    import joblib
    return joblib.load(path)


def open_source(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        # DirectShow opens webcams much faster on Windows
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise SystemExit("Couldn't open the camera/video. Try --camera 1, or --video path/to/file.mp4")
    return cap


# ---------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default=0, type=int)
    ap.add_argument("--video", default=None, help="Use a video file instead of the webcam")
    ap.add_argument("--mode", default="classify", choices=["classify", "pca"])
    ap.add_argument("--probe", default="results/probe.joblib", type=Path)
    ap.add_argument("--model", default="small", choices=["small", "base", "large"],
                    help="Backbone for PCA mode (classify mode uses the probe's backbone)")
    ap.add_argument("--img-size", default=224, type=int,
                    help="Bigger = finer PCA detail but slower (must be a multiple of 14)")
    ap.add_argument("--display-size", default=480, type=int)
    args = ap.parse_args()

    probe = load_probe(args.probe)
    if probe is None:
        print(f"No probe at {args.probe} (run train_probe.py first). Only PCA mode available.")
        args.mode = "pca"
    model_size = probe["dino_model"] if probe else args.model
    if probe and probe["img_size"] != args.img_size:
        print(f"Note: probe was trained at {probe['img_size']}px; classify mode may be less accurate.")

    model, device = load_dino(model_size)
    tf = make_transform(args.img_size)
    grid = args.img_size // PATCH_SIZE
    live_pca = LivePCA(grid)
    cap = open_source(args)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print(f"DINOv2-{model_size} on {device}. Keys: M mode, R refit colors, S screenshot, Q quit")
    fps, last = 0.0, time.time()
    D = args.display_size

    while True:
        ok, frame = cap.read()
        if not ok:
            if args.video:  # loop the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        square = center_square(frame)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        batch = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
        cls, patches = dino_features(model, batch)

        view = cv2.resize(square, (D, D))

        if args.mode == "classify":
            probs = probe["model"].predict_proba(cls.cpu().numpy())[0]
            top = probs.argsort()[::-1][:3]
            for rank, i in enumerate(top):
                name = probe["classes"][i].replace("_", " ")
                color = (0, 255, 0) if rank == 0 else (230, 230, 230)
                put_text(view, f"{name}: {probs[i]:.0%}", (12, 64 + 30 * rank), 0.75 if rank == 0 else 0.6, color)
            canvas = view
        else:
            pca_rgb = live_pca.apply(patches[0].cpu().numpy())
            pca_big = cv2.resize(cv2.cvtColor(pca_rgb, cv2.COLOR_RGB2BGR), (D, D),
                                 interpolation=cv2.INTER_NEAREST)
            put_text(pca_big, "DINO patch features -> PCA -> RGB", (12, D - 16), 0.5)
            canvas = np.hstack([view, pca_big])

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        put_text(canvas, f"{args.mode.upper()}  |  {fps:4.1f} FPS  |  M mode  R refit  S save  Q quit",
                 (12, 28), 0.55)

        cv2.imshow("DINOv2 live demo", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m") and probe is not None:
            args.mode = "pca" if args.mode == "classify" else "classify"
        if key == ord("r"):
            live_pca.fit(patches[0].cpu().numpy())
        if key == ord("s"):
            path = out_dir / f"webcam_{args.mode}_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(str(path), canvas)
            print(f"Saved {path}")
        if cv2.getWindowProperty("DINOv2 live demo", cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed with the X button

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import torch
    torch.set_grad_enabled(False)
    main()
