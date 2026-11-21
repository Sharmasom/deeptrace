"""Phase 1.5: frames -> faces -> tracked 224x224 crops, resumable and backed up to Drive.

Per video:
  1. read sequentially, keep every STRIDE-th frame until NUM_FRAMES (first ~10-13 s);
     short videos are sampled uniformly over their whole length
  2. MTCNN on every sampled frame, all faces (keep_all), on a downscaled copy
  3. greedy IoU tracking across frames -> face tracks; gaps filled from the nearest
     detected frame of the same track
  4. up to MAX_TRACKS tracks kept (those seen in >= MIN_TRACK_FRAC of frames),
     ordered by mean face area; track 0 = largest face (the old single-face behaviour)
  5. square crop with MARGIN, resize to IMG_SIZE, JPEG q95; 5-point landmarks are
     stored in crop coordinates (used later for mouth boxes, E2)

Output: LOCAL/crops/<dataset>/<rel without ext>/t<k>_<i>.jpg + meta.json
Every SHARD_SIZE finished videos are tarred to Drive (paths.SHARDS) and listed in
done_<dataset>.txt, so a dropped Colab session loses at most one shard of work.
"""
import json, tarfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from deeptrace import paths

NUM_FRAMES, STRIDE, IMG_SIZE, MARGIN = 32, 10, 224, 1.3
DET_MAX_SIDE, DET_THRESH = 640, 0.90
MAX_TRACKS, MIN_TRACK_FRAC, IOU_LINK = 3, 0.5, 0.3
JPEG_Q, SHARD_SIZE = 95, 400
TAG = ""          # set per Colab worker ('A'/'B') so parallel workers never collide on Drive


# ------------------------------------------------------------------ frames
def sample_frames(path, n=NUM_FRAMES, stride=STRIDE):
    """Read the video sequentially from the start and keep every `stride`-th frame
    (n frames = a ~10-13 s window at 25-30 fps). Sequential grab() avoids the
    expensive per-frame seeks of 1080p H.264, and is also robust to VFR files and
    broken frame counts. Short videos: stride shrinks so n frames still span them."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if 0 < total < n * stride:
        want = set(np.linspace(0, total - 1, n).round().astype(int).tolist())
    else:
        want = set(range(0, n * stride, stride))
    last = max(want)
    frames, idxs, i = [], [], 0
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, fr = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); idxs.append(i)
        i += 1
    cap.release()
    return frames, idxs, [round(j / fps, 3) for j in idxs]


# ------------------------------------------------------------------ tracking
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def track(dets, n_frames):
    """dets[t] = list of (box[4], landmarks[5,2], prob). Returns list of tracks,
    each a dict t -> (box, lm). Greedy: link to the track whose last box has the
    highest IoU above IOU_LINK; otherwise start a new track."""
    tracks = []
    for t in range(n_frames):
        used = set()
        for box, lm, p in sorted(dets[t], key=lambda d: -(d[0][2] - d[0][0]) * (d[0][3] - d[0][1])):
            best, best_iou = None, IOU_LINK
            for k, tr in enumerate(tracks):
                if k in used or t in tr:
                    continue
                last = tr[max(tr)][0]
                v = iou(box, last)
                if v > best_iou:
                    best, best_iou = k, v
            if best is None:
                tracks.append({t: (box, lm)}); used.add(len(tracks) - 1)
            else:
                tracks[best][t] = (box, lm); used.add(best)
    keep = [tr for tr in tracks if len(tr) >= MIN_TRACK_FRAC * n_frames]
    if not keep and tracks:                   # nothing stable: keep the longest track
        keep = [max(tracks, key=len)]
    area = lambda tr: np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b, _ in tr.values()])
    return sorted(keep, key=area, reverse=True)[:MAX_TRACKS]


def fill(tr, n_frames):
    """Fill missing frames with the nearest detected frame's box/landmarks."""
    have = sorted(tr)
    out, filled = [], 0
    for t in range(n_frames):
        if t in tr:
            out.append(tr[t])
        else:
            nearest = min(have, key=lambda s: abs(s - t))
            out.append(tr[nearest]); filled += 1
    return out, filled


def crop(frame, box, lm):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * MARGIN / 2
    X1, Y1 = int(round(cx - half)), int(round(cy - half))
    side = int(round(2 * half))
    # pad with edge replication so faces near the border keep a square, centred crop
    pad = max(0, -X1, -Y1, X1 + side - w, Y1 + side - h)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    face = frame[Y1 + pad:Y1 + pad + side, X1 + pad:X1 + pad + side]
    s = IMG_SIZE / side
    lm_c = (np.asarray(lm) - [X1, Y1]) * s
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA), lm_c


# ------------------------------------------------------------------ per video
class Extractor:
    def __init__(self, device=None):
        import torch
        from facenet_pytorch import MTCNN
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=self.dev, post_process=False,
                           select_largest=False, thresholds=[0.6, 0.7, 0.7])

    def detect(self, frames):
        h, w = frames[0].shape[:2]
        s = min(1.0, DET_MAX_SIDE / max(h, w))
        small = [cv2.resize(f, (round(w * s), round(h * s))) for f in frames] if s < 1 else frames
        boxes, probs, lms = self.mtcnn.detect(np.stack(small), landmarks=True)
        dets = []
        for b, p, l in zip(boxes, probs, lms):
            cur = []
            if b is not None:
                for bb, pp, ll in zip(b, p, l):
                    # MTCNN returns object arrays when face counts differ across frames
                    if pp is not None and float(pp) >= DET_THRESH:
                        cur.append((np.asarray(bb, dtype=np.float64) / s,
                                    np.asarray(ll, dtype=np.float64) / s, float(pp)))
            dets.append(cur)
        return dets

    def process(self, sampled, out_dir):
        frames, idxs, ts = sampled
        meta = dict(n_sampled=len(frames), frame_idx=idxs, timestamps=ts, tracks=[])
        if len(frames) < NUM_FRAMES // 2:
            meta["status"] = "unreadable"
            return meta
        while len(frames) < NUM_FRAMES:            # short videos: repeat last frame
            frames.append(frames[-1]); idxs.append(idxs[-1]); ts.append(ts[-1])
        dets = self.detect(frames)
        meta["faces_per_frame"] = [len(d) for d in dets]
        tracks = track(dets, NUM_FRAMES)
        if not tracks:
            meta["status"] = "no_face"
            return meta
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, tr in enumerate(tracks):
            seq, filled = fill(tr, NUM_FRAMES)
            lms, boxes = [], []
            for i, (fr, (box, lm)) in enumerate(zip(frames, seq)):
                face, lm_c = crop(fr, box, lm)
                cv2.imwrite(str(out_dir / f"t{k}_{i:02d}.jpg"), cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                lms.append(np.round(lm_c, 1).tolist()); boxes.append(np.round(box, 1).tolist())
            meta["tracks"].append(dict(k=k, detected=len(tr), filled=filled, boxes=boxes, landmarks=lms))
        meta["status"] = "ok"
        (out_dir / "meta.json").write_text(json.dumps(meta))
        return meta


# ------------------------------------------------------------------ resumable driver
def _done_file(dataset):
    return paths.SHARDS / (f"done_{dataset}_{TAG}.txt" if TAG else f"done_{dataset}.txt")


def restore(dataset):
    """Unpack every Drive shard of this dataset onto the local disk; return finished rels."""
    root = paths.CROPS / dataset
    root.mkdir(parents=True, exist_ok=True)
    for sh in sorted(paths.SHARDS.glob(f"{dataset}_*.tar")):
        marker = root / f".unpacked_{sh.name}"
        if not marker.exists():
            with tarfile.open(sh) as t:
                t.extractall(root)
            marker.touch()
    done = set()
    for f in paths.SHARDS.glob(f"done_{dataset}*.txt"):        # all workers
        done |= set(f.read_text().splitlines())
    st = status_table(dataset)
    if len(st):                       # videos that errored are retried, never counted as done
        last = st.drop_duplicates("rel", keep="last")
        done -= set(last.loc[last.status.str.startswith("error"), "rel"])
    return done


def _flush(dataset, rels, results):
    if not rels:
        return
    root = paths.CROPS / dataset
    pat = f"{dataset}_{TAG}[0-9]*.tar" if TAG else f"{dataset}_[0-9]*.tar"
    n = len(list(paths.SHARDS.glob(pat)))
    tmp = paths.LOCAL / f"{dataset}_{TAG}{n:04d}.tar"
    with tarfile.open(tmp, "w") as t:
        for r in rels:
            d = root / Path(r).with_suffix("")
            if d.exists():
                t.add(d, arcname=str(Path(r).with_suffix("")))
    status = paths.SHARDS / f"status_{dataset}_{TAG}{n:04d}.json"
    status.write_text(json.dumps(results))
    dest = paths.SHARDS / tmp.name
    import shutil
    shutil.copy(tmp, str(dest) + ".part"); Path(str(dest) + ".part").rename(dest)   # atomic-ish on Drive
    tmp.unlink()
    (root / f".unpacked_{dest.name}").touch()
    with open(_done_file(dataset), "a") as f:
        f.write("\n".join(rels) + "\n")


def run(df, dataset, root, limit=None, workers=None):
    import os
    workers = workers or max(2, min(8, (os.cpu_count() or 2)))
    """df: inventory rows of one dataset (needs 'rel'). root: dataset root on disk."""
    done = restore(dataset)
    todo = [r for r in df.rel if r not in done][: limit or None]
    print(f"[{dataset}] done {len(done)}, to do {len(todo)}")
    if not todo:
        return
    ex = Extractor()
    pending, results, t0 = [], {}, time.time()
    def prefetch(pool, depth=2 * workers):
        # bounded look-ahead: at most `depth` decoded videos held in RAM
        from collections import deque
        q, it = deque(), iter(todo)
        for r in it:
            q.append((r, pool.submit(sample_frames, Path(root) / r)))
            if len(q) >= depth:
                break
        while q:
            r, fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt, pool.submit(sample_frames, Path(root) / nxt)))
            try:
                yield r, fut.result()
            except Exception:
                yield r, ([], [], [])

    with ThreadPoolExecutor(workers) as pool:
        for i, (rel, sampled) in enumerate(prefetch(pool)):
            try:
                m = ex.process(sampled, paths.CROPS / dataset / Path(rel).with_suffix(""))
                results[rel] = dict(status=m["status"], n_tracks=len(m["tracks"]),
                                    faces_max=max(m.get("faces_per_frame", [0]) or [0]),
                                    detected=m["tracks"][0]["detected"] if m["tracks"] else 0)
            except Exception as e:                      # never lose the batch to one bad file
                results[rel] = dict(status=f"error: {e}", n_tracks=0, faces_max=0, detected=0)
            pending.append(rel)
            if len(pending) >= SHARD_SIZE:
                _flush(dataset, pending, {r: results[r] for r in pending}); pending = []
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{len(todo)}  {rate:.2f} vid/s  eta {(len(todo)-i-1)/rate/60:.0f} min")
    _flush(dataset, pending, {r: results[r] for r in pending})
    errs = {r: v["status"] for r, v in results.items() if v["status"].startswith("error")}
    if errs:
        print(f"[{dataset}] {len(errs)} videos FAILED (will be retried on the next run):", list(errs.items())[:5])
    print(f"[{dataset}] finished {len(todo)} in {(time.time()-t0)/60:.1f} min")


def status_table(dataset):
    rows = {}
    for f in sorted(paths.SHARDS.glob(f"status_{dataset}_*.json")):
        rows.update(json.loads(f.read_text()))          # later shards overwrite earlier attempts
    import pandas as pd
    if not rows:
        return pd.DataFrame(columns=["rel", "status", "n_tracks", "faces_max", "detected"])
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("rel").reset_index()
