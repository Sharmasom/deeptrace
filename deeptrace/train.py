"""Training/evaluation for one run, fully resumable from Drive.

run(cfg) where cfg is a dict, e.g.
  dict(name="v1_mean", train_on="ffpp", head="mean", freq=False, seed=0,
       epochs=8, bs=8, T=16, view="full", aug=dict(p_jpeg=0.3, p_blur=0.2, p_down=0.0),
       methods=None)          # FF++ method subset for training (LOMO); None = all

Layout on Drive: runs/<run_id>/ config.json, history.csv, last.pt, best.pt,
                 preds_<set>.csv, metrics.json  (run_id = name_trainon_s<seed>)
Re-running the same cfg resumes from last.pt, or only re-evaluates if finished.
"""
import json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deeptrace import paths
from deeptrace.data.dataset import ClipDS, available
from deeptrace.evaluate import metrics, youden, per_method_auc
from deeptrace.models.net import Net

DEFAULTS = dict(head="mean", freq=False, seed=0, epochs=8, bs=8, T=16, view="full",
                aug=dict(p_jpeg=0.3, p_blur=0.2, p_down=0.0), methods=None, smoke=False, lr_bb=1e-4, lr_head=1e-3,
                unfreeze_from=5, workers=4, test_view=None, limit=None)

# training set -> (train/val manifest, test sets evaluated after training)
PROTOCOL = {
    "ffpp": ("ffpp", ["ffpp", "celebdf_strict", "celebdf_official"]),
    "celebdf_strict": ("celebdf_strict", ["celebdf_strict", "ffpp"]),
    "celebdf_official": ("celebdf_official", ["celebdf_official", "ffpp"]),
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


FFPP_METHODS = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]   # D9: standard FF++ 4 methods


def load_split(name):
    d = pd.read_csv(paths.MANIFESTS / f"split_{name}.csv")
    d = d[d.split.isin(["train", "val", "test"])]
    if name == "ffpp":
        d = d[d.method.isin(FFPP_METHODS)]
    return d


def loader(df, train, cfg, view=None):
    ds = ClipDS(df, train, T=cfg["T"], view=view or cfg["view"], aug=cfg["aug"])
    return DataLoader(ds, batch_size=cfg["bs"], shuffle=train, drop_last=train, num_workers=cfg["workers"],
                      pin_memory=True, persistent_workers=False)


@torch.no_grad()
def predict(model, dl):
    model.eval(); P = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            P.append(torch.sigmoid(model(x.cuda(non_blocking=True)).float()).cpu())
    return torch.cat(P).numpy()


def run(cfg):
    cfg = {**DEFAULTS, **cfg}
    run_id = f"{cfg['name']}_{cfg['train_on']}_s{cfg['seed']}"
    out = paths.RUNS / run_id; out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() and not cfg.get("force"):
        print(f"[{run_id}] already finished - skipped (set force=True to redo)")
        return json.loads((out / "metrics.json").read_text())
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    set_seed(cfg["seed"])
    split_name, test_sets = PROTOCOL[cfg["train_on"]]
    sp = load_split(split_name)
    if cfg.get("smoke"):            # code check only: any available crops, re-dealt into train/val/test
        sp = available(sp.assign(split="x"))
        sp["split"] = [("train", "val", "test")[i % 3] for i in sp.groupby("label").cumcount()]
        test_sets = ["smoke"]
    tr, va = sp[sp.split == "train"], sp[sp.split == "val"]
    if cfg["methods"]:                                 # LOMO: restrict fakes in train AND val
        keep = lambda d: d[(d.label == 0) | d.method.isin(cfg["methods"])]
        tr, va = keep(tr), keep(va)
    tr, va = available(tr), available(va)
    if cfg["limit"]:                                   # smoke test only: tiny balanced subset
        sub = lambda d: d.groupby("label", group_keys=False).head(cfg["limit"])
        tr, va = sub(tr), sub(va)
    print(f"[{run_id}] train {len(tr)} ({int(tr.label.sum())} fake) | val {len(va)} ({int(va.label.sum())} fake)")

    model = Net(cfg["head"], cfg["freq"], cfg["unfreeze_from"]).cuda()
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": cfg["lr_bb"]}, {"params": rest, "lr": cfg["lr_head"]}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda")
    pos_w = torch.tensor((tr.label == 0).sum() / max(1, (tr.label == 1).sum()), dtype=torch.float32).cuda()
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)       # down-weights the majority (fake) class

    start, best, hist = 0, -1.0, []
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start, best, hist = ck["epoch"] + 1, ck["best"], ck["hist"]
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
        print(f"  resumed at epoch {start}")

    tl, vl = loader(tr, True, cfg), loader(va, False, cfg)
    from sklearn.metrics import roc_auc_score
    for ep in range(start, cfg["epochs"]):
        model.train(); t0, tot, n = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item(); n += 1
        sched.step()
        pv = predict(model, vl); auc = roc_auc_score(va.label, pv)
        hist.append(dict(epoch=ep + 1, loss=tot / max(n, 1), val_auc=auc, sec=time.time() - t0))
        print(f"  ep{ep+1}/{cfg['epochs']} loss {tot/max(n,1):.4f} val_auc {auc:.4f} ({time.time()-t0:.0f}s)")
        if auc > best:
            best = auc; torch.save(model.state_dict(), out / "best.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=ep, best=best, hist=hist, py_rng=random.getstate(),
                        np_rng=np.random.get_state(), torch_rng=torch.get_rng_state()), out / "last.pt")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    # ---------------- evaluation: threshold from validation only, then every test set
    model.load_state_dict(torch.load(out / "best.pt", map_location="cuda"))
    tv = cfg["test_view"] or cfg["view"]
    pv = predict(model, loader(va, False, cfg, tv)); thr = youden(va.label, pv)
    va.assign(p=pv)[["dataset", "rel", "label", "method", "p"]].to_csv(out / "preds_val.csv", index=False)
    res = dict(run_id=run_id, val=metrics(va.label, pv, thr), best_val_auc=best,
               params_M=sum(p.numel() for p in model.parameters()) / 1e6)
    for ts in test_sets:
        d = sp if ts == "smoke" else load_split(ts)
        d = available(d[d.split == "test"])
        if cfg["limit"]:
            d = d.groupby("label", group_keys=False).head(cfg["limit"])
        p = predict(model, loader(d, False, cfg, tv))
        d = d.assign(p=p)
        d[["dataset", "rel", "label", "method", "p"]].to_csv(out / f"preds_{ts}.csv", index=False)
        res[ts] = metrics(d.label, p, thr)
        if ts == "ffpp":
            res[ts]["per_method_auc"] = per_method_auc(d)
        print(f"  test {ts:17s} AUC {res[ts].get('auc', float('nan')):.4f}  bal_acc {res[ts]['bal_acc']:.3f}  "
              f"real_rec {res[ts]['real_rec']:.3f} fake_rec {res[ts]['fake_rec']:.3f}")
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    del model; torch.cuda.empty_cache()
    return res


def collect():
    """All finished runs -> one tidy table (one row per run x test set)."""
    rows = []
    for f in sorted(paths.RUNS.glob("*/metrics.json")):
        m = json.loads(f.read_text()); cfg = json.loads((f.parent / "config.json").read_text())
        for k, v in m.items():
            if isinstance(v, dict) and "acc" in v and k != "val":
                rows.append(dict(run=m["run_id"], name=cfg["name"], train_on=cfg["train_on"], seed=cfg["seed"],
                                 test=k, auc=v.get("auc"), bal_acc=v["bal_acc"], real_rec=v["real_rec"],
                                 fake_rec=v["fake_rec"], params_M=m["params_M"]))
    return pd.DataFrame(rows)
