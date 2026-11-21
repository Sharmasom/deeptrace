"""Evaluation-only experiments on trained checkpoints (no training).

degradation(run_id)  G5: crop-level JPEG / downscale on test crops, threshold fixed from clean val
mouth_eval(run_id)   G1: test a checkpoint under full / mouth_masked / mouth_only views, overall,
                     per FF++ method and per mouth-motion tercile (motion = std over time of the
                     MTCNN mouth-corner distance, normalised by eye distance)
timing(run_id)       G7: ms per 16-frame clip on GPU and CPU, parameters
"""
import json, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from deeptrace import paths
from deeptrace.data import dataset as D
from deeptrace.data.dataset import ClipDS, available
from deeptrace.evaluate import metrics, per_method_auc
from deeptrace.models.net import Net
from deeptrace.train import load_split, predict, PROTOCOL


def load_model(run_id):
    cfg = json.loads((paths.RUNS / run_id / "config.json").read_text())
    m = Net(cfg["head"], cfg["freq"], cfg["unfreeze_from"]).cuda()
    m.load_state_dict(torch.load(paths.RUNS / run_id / "best.pt", map_location="cuda"))
    thr = json.loads((paths.RUNS / run_id / "metrics.json").read_text())["val"]["threshold"]
    return m.eval(), cfg, thr


class DegDS(ClipDS):
    def __init__(self, df, q=None, down=None, **kw):
        super().__init__(df, False, **kw); self.q, self.down = q, down

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        if not (self.q or self.down):
            return x, y
        fr = []
        for f in x.permute(0, 2, 3, 1).numpy():
            if self.down:
                f = cv2.resize(cv2.resize(f, (self.down, self.down), interpolation=cv2.INTER_AREA), (224, 224))
            if self.q:
                _, e = cv2.imencode(".jpg", np.ascontiguousarray(f[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, self.q])
                f = cv2.imdecode(e, 1)[:, :, ::-1]
            fr.append(np.ascontiguousarray(f))
        return torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2), y


CONDS = {"clean": {}, "jpeg50": dict(q=50), "jpeg30": dict(q=30), "down112": dict(down=112),
         "down64": dict(down=64), "jpeg50+down112": dict(q=50, down=112)}


def degradation(run_id, test_sets=None, bs=16):
    model, cfg, thr = load_model(run_id)
    test_sets = test_sets or PROTOCOL[cfg["train_on"]][1][:2]
    rows = []
    for ts in test_sets:
        d = available(load_split(ts).query("split == 'test'"))
        for name, kw in CONDS.items():
            p = predict(model, DataLoader(DegDS(d, **kw), batch_size=bs, num_workers=4))
            m = metrics(d.label, p, thr)
            rows.append(dict(run=run_id, test=ts, cond=name, auc=m["auc"], bal_acc=m["bal_acc"],
                             real_rec=m["real_rec"], fake_rec=m["fake_rec"]))
            print(run_id, ts, name, round(m["auc"], 4))
    del model; torch.cuda.empty_cache()
    out = pd.DataFrame(rows); out.to_csv(paths.RUNS / run_id / "degradation.csv", index=False)
    return out


def mouth_motion(df):
    vals = []
    for r in df.itertuples():
        m = D._meta(str(paths.CROPS / r.dataset / Path(r.rel).with_suffix("")))
        lm = np.array(m["tracks"][0]["landmarks"])
        mouth = np.linalg.norm(lm[:, 3] - lm[:, 4], axis=1); eyes = np.linalg.norm(lm[:, 0] - lm[:, 1], axis=1)
        vals.append(float(np.std(mouth / np.maximum(eyes, 1e-3))))
    return np.array(vals)


def mouth_eval(run_id, test_set="ffpp", bs=16):
    model, cfg, thr = load_model(run_id)
    d = available(load_split(test_set).query("split == 'test'"))
    d = d.assign(motion=mouth_motion(d))
    d["motion_group"] = pd.qcut(d.motion, 3, labels=["low", "mid", "high"])
    rows = []
    for view in ["full", "mouth_masked", "mouth_only"]:
        p = predict(model, DataLoader(ClipDS(d, False, view=view), batch_size=bs, num_workers=4))
        dd = d.assign(p=p)
        m = metrics(dd.label, p, thr)
        row = dict(run=run_id, trained_view=cfg["view"], test_view=view, test=test_set, auc=m["auc"], bal_acc=m["bal_acc"])
        if test_set == "ffpp":
            row.update({f"auc_{k}": v for k, v in per_method_auc(dd).items()})
        for g, sub in dd.groupby("motion_group", observed=True):
            if sub.label.nunique() == 2:
                from sklearn.metrics import roc_auc_score
                row[f"auc_motion_{g}"] = roc_auc_score(sub.label, sub.p); row[f"n_motion_{g}"] = len(sub)
        rows.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    del model; torch.cuda.empty_cache()
    out = pd.DataFrame(rows); out.to_csv(paths.RUNS / run_id / f"mouth_eval_{test_set}.csv", index=False)
    return out


@torch.no_grad()
def timing(run_id, n=20):
    model, cfg, _ = load_model(run_id)
    x = torch.randint(0, 255, (1, 16, 3, 224, 224), dtype=torch.uint8)
    res = dict(run=run_id, params_M=sum(p.numel() for p in model.parameters()) / 1e6)
    xg = x.cuda()
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.float16): model(xg)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.float16): model(xg)
    torch.cuda.synchronize(); res["gpu_ms_per_clip"] = (time.time() - t) / n * 1000
    mc = model.float().cpu(); t = time.time()
    for _ in range(3): mc(x)
    res["cpu_ms_per_clip"] = (time.time() - t) / 3 * 1000
    del model, mc; torch.cuda.empty_cache()
    return res
