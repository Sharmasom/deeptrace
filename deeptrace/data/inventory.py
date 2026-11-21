"""Phase 1.2: build a verified inventory of both datasets.

Each row is one video: dataset, rel path, label (fake=1), method, identity
fields used for leakage-safe splitting, plus technical metadata (fps, frames,
resolution, duration, size, hash) for profiling and duplicate detection.
"""
import hashlib
import re
from pathlib import Path

import cv2
import pandas as pd
from tqdm.auto import tqdm

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
FF_METHODS = {"deepfakes": "Deepfakes", "face2face": "Face2Face", "faceswap": "FaceSwap",
              "neuraltextures": "NeuralTextures", "faceshifter": "FaceShifter",
              "deepfakedetection": "DeepFakeDetection"}
FF_REAL_HINTS = ("original", "youtube", "actors", "real")


def tree(root, depth=3, max_per_dir=6):
    """Print a compact directory tree so the dataset layout is visible before parsing."""
    root = Path(root)
    def walk(d, lvl):
        if lvl > depth:
            return
        kids = sorted(d.iterdir())
        dirs = [k for k in kids if k.is_dir()]
        files = [k for k in kids if k.is_file()]
        print("  " * lvl + f"{d.name}/  ({len(dirs)} dirs, {len(files)} files)")
        for f in files[:max_per_dir]:
            print("  " * (lvl + 1) + f.name)
        for k in dirs:
            walk(k, lvl + 1)
    walk(root, 0)


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) if ok else 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok else 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok else 0
    readable = False
    if ok and n > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        readable = cap.read()[0]
    cap.release()
    return dict(fps=round(fps, 3), n_frames=n, width=w, height=h,
                duration_s=round(n / fps, 2) if fps > 0 else 0.0, readable=bool(readable))


def file_hash(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- Celeb-DF v2
def celebdf_rows(root):
    root = Path(root)
    rows = []
    for sub, label in [("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)]:
        for p in sorted((root / sub).glob("*.mp4")):
            ids = re.findall(r"id\d+", p.stem)
            rows.append(dict(dataset="celebdf", rel=f"{sub}/{p.name}", path=str(p), label=label,
                             method="CelebDF-swap" if label else sub,
                             id_a=ids[0] if ids else f"yt_{p.stem}",
                             id_b=ids[1] if len(ids) > 1 else (ids[0] if ids else f"yt_{p.stem}"),
                             has_identity=bool(ids)))
    return rows


def celebdf_official_test(root):
    f = Path(root) / "List_of_testing_videos.txt"
    out = {}
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1].strip()] = 1 - int(parts[0])   # file uses 1=real, 0=fake
    return out


# ---------------------------------------------------------------- FF++ c23
def ff_rows(root):
    """Parse FF++ wherever it sits under root. Method = path component naming a
    manipulation; real = path containing original/youtube/actors.
    Standard fakes are named <target>_<source>.mp4; originals <id>.mp4."""
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        parts = [s.lower() for s in p.relative_to(root).parts[:-1]]
        method = next((FF_METHODS[s] for s in parts if s in FF_METHODS), None)
        is_real = method is None and any(h in s for s in parts for h in FF_REAL_HINTS)
        if method is None and not is_real:
            continue                      # unknown folder: reported, not guessed
        stem = p.stem
        m = re.fullmatch(r"(\d{3})_(\d{3})", stem)
        if m:
            id_a, id_b = m.group(1), m.group(2)          # target, source
        elif re.fullmatch(r"\d{3}", stem):
            id_a = id_b = stem
        else:                                            # DFD actor videos: 01__scene / 01_02__scene__X
            a = re.match(r"(\d{2})(?:_(\d{2}))?__", stem)
            id_a = f"act{a.group(1)}" if a else stem
            id_b = f"act{a.group(2)}" if a and a.group(2) else id_a
        rows.append(dict(dataset="ffpp", rel=str(p.relative_to(root)), path=str(p),
                         label=0 if is_real else 1,
                         method="original" if is_real else method,
                         id_a=id_a, id_b=id_b, has_identity=True))
    return rows


def build(rows, hash_files=True, desc="profile"):
    df = pd.DataFrame(rows)
    meta = [video_meta(p) for p in tqdm(df.path, desc=desc)]
    df = pd.concat([df, pd.DataFrame(meta)], axis=1)
    df["size_mb"] = [round(Path(p).stat().st_size / 1e6, 3) for p in df.path]
    if hash_files:
        df["md5"] = [file_hash(p) for p in tqdm(df.path, desc="hash")]
    return df


def summarize(df):
    print("\nvideos by dataset / method / label")
    print(df.groupby(["dataset", "method", "label"]).size().to_string())
    print("\nunreadable:", int((~df.readable).sum()))
    print("\nresolution (top):")
    print(df.groupby(["dataset", "width", "height"]).size().sort_values(ascending=False).head(10).to_string())
    print("\nfps:")
    print(df.groupby("dataset").fps.describe().round(2).to_string())
    print("\nduration s:")
    print(df.groupby("dataset").duration_s.describe().round(1).to_string())
    if "md5" in df:
        dup = df[df.duplicated("md5", keep=False)].sort_values("md5")
        print(f"\nexact duplicate files: {len(dup)} rows in {dup.md5.nunique()} groups")
        if len(dup):
            print(dup[["dataset", "rel", "label", "md5"]].head(20).to_string())
            mixed = dup.groupby("md5").label.nunique()
            print("duplicate groups with conflicting labels:", int((mixed > 1).sum()))
