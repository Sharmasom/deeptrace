"""Clip dataset over extracted crops.

A sample is one face track of one video: T frames (default 16 of the 32 saved,
every other frame; random offset 0/1 in training) -> uint8 tensor [T,3,224,224].
Augmentations are drawn once per clip and applied identically to all its frames,
so temporal models do not see artificial frame-to-frame flicker.

Views (for E2, mouth evidence): full | mouth_masked | mouth_only, using the
MTCNN mouth-corner landmarks stored in meta.json (indices 3, 4).
"""
import json, random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from deeptrace import paths

MASK_GREY = (124, 116, 104)        # ImageNet mean in RGB uint8 -> ~0 after normalisation


@lru_cache(maxsize=None)
def _meta(d):
    f = Path(d) / "meta.json"
    return json.loads(f.read_text()) if f.exists() else None


def mouth_box(meta, track=0, scale=1.9):
    """Square box around the mouth from the mouth-corner landmarks (median over frames)."""
    lm = np.array(meta["tracks"][track]["landmarks"])         # [32,5,2]
    c = lm[:, 3:5].mean(1)                                     # mouth centre per frame
    w = np.linalg.norm(lm[:, 3] - lm[:, 4], axis=1)            # mouth width per frame
    cx, cy = np.median(c, 0); side = float(np.clip(np.median(w) * scale, 48, 140))
    return int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)


# ------------------------------------------------------------------ augmentation
def augment(frames, cfg):
    """frames: list of HxWx3 uint8 RGB. Same random parameters for every frame."""
    if random.random() < 0.5:
        frames = [f[:, ::-1] for f in frames]
    a, b = random.uniform(0.8, 1.2), random.uniform(-20, 20)
    frames = [np.clip(f.astype(np.float32) * a + b, 0, 255).astype(np.uint8) for f in frames]
    if random.random() < cfg.get("p_blur", 0.2):
        k = random.choice([3, 5])
        frames = [cv2.GaussianBlur(np.ascontiguousarray(f), (k, k), 0) for f in frames]
    if random.random() < cfg.get("p_down", 0.0):          # resolution robustness (G5)
        s = random.choice([56, 64, 80, 112, 160])
        frames = [cv2.resize(cv2.resize(np.ascontiguousarray(f), (s, s), interpolation=cv2.INTER_AREA),
                             (224, 224), interpolation=cv2.INTER_LINEAR) for f in frames]
    if random.random() < cfg.get("p_jpeg", 0.3):          # compression robustness (G5)
        q = random.randint(cfg.get("jpeg_min", 40), 90)
        out = []
        for f in frames:
            _, enc = cv2.imencode(".jpg", np.ascontiguousarray(f[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, q])
            out.append(cv2.imdecode(enc, 1)[:, :, ::-1])
        frames = out
    return [np.ascontiguousarray(f) for f in frames]


def apply_view(frames, meta, view):
    if view == "full":
        return frames
    x1, y1, x2, y2 = mouth_box(meta)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, 224), min(y2, 224)
    if view == "mouth_only":
        return [cv2.resize(np.ascontiguousarray(f[y1:y2, x1:x2]), (224, 224)) for f in frames]
    if view == "mouth_masked":
        out = []
        for f in frames:
            f = f.copy(); f[y1:y2, x1:x2] = MASK_GREY; out.append(f)
        return out
    raise ValueError(view)


# ------------------------------------------------------------------ dataset
class ClipDS(Dataset):
    """df rows need: dataset, rel, label. Videos without crops are filtered out
    by `available()` before building the dataset (and counted, never silently)."""

    def __init__(self, df, train, T=16, view="full", aug=None, track=0):
        self.df = df.reset_index(drop=True)
        self.train, self.T, self.view, self.aug, self.track = train, T, view, aug or {}, track

    def __len__(self):
        return len(self.df)

    def dir(self, r):
        return paths.CROPS / r.dataset / Path(r.rel).with_suffix("")

    def __getitem__(self, i):
        r = self.df.iloc[i]
        d = self.dir(r)
        off = random.randint(0, 1) if self.train else 0
        idx = list(range(off, 32, 32 // self.T))[: self.T]
        frames = [cv2.cvtColor(cv2.imread(str(d / f"t{self.track}_{j:02d}.jpg")), cv2.COLOR_BGR2RGB) for j in idx]
        frames = apply_view(frames, _meta(str(d)), self.view)
        if self.train:
            frames = augment(frames, self.aug)
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)      # T,3,H,W uint8
        return x, torch.tensor(float(r.label))


def available(df):
    """Keep rows whose track-0 crops exist; report what was dropped."""
    ok = np.array([(paths.CROPS / r.dataset / Path(r.rel).with_suffix("") / "t0_31.jpg").exists()
                   for r in df.itertuples()])
    if (~ok).any():
        print(f"  [data] {int((~ok).sum())} videos have no face crops and are excluded:",
              df[~ok].groupby(["split", "label"]).size().to_dict())
    return df[ok].reset_index(drop=True)
