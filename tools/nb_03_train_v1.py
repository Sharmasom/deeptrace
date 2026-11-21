from make_nb import md, code

HEADER = md("""# DeepTrace 03 - Model v1 baseline (Phase 2)
Runtime: **T4 GPU**. Cells 2-3 (restore + smoke test) can run while notebook 02 is still extracting.
Cell 4 refuses to start until extraction is complete. Every run is resumable: after a disconnect, Run all again.
Model v1 = EfficientNet-B0 (blocks 5+ fine-tuned) + mean over 16 frames. 3 seeds x 2 training sets.""")

CELLS = [
code('''# Cell 2 - restore face crops from the Drive shards onto this runtime's disk
import time, json, pandas as pd, torch
from deeptrace import paths
from deeptrace.data import prep
t0 = time.time()
for ds in ["celebdf", "ffpp"]:
    done = prep.restore(ds)
    print(ds, "videos with crops:", len(done))
print(f"restored in {time.time()-t0:.0f}s | GPU: {torch.cuda.get_device_name(0)}")
'''),
code('''# Cell 3 - smoke test of every architecture on a few videos (1 epoch). Catches bugs before real runs.
from deeptrace import train as T
for head, freq in [("mean", False), ("bilstm", False), ("tdiff", False), ("mean", True)]:
    r = T.run(dict(name=f"smoke_{head}{'_freq' if freq else ''}", train_on="ffpp", head=head, freq=freq,
                   epochs=1, smoke=True, workers=2, force=True))
    print(head, freq, "params M:", round(r["params_M"], 2))
print("SMOKE TEST PASSED")
'''),
code('''# Cell 4 - Phase 2: Model v1, 3 seeds on FF++ and on strict Celeb-DF (~1.5 h). Resumable.
need = {"celebdf": 6529, "ffpp": 6000}
have = {ds: len(prep.restore(ds)) for ds in need}
missing = {ds: need[ds] - have[ds] for ds in need if have[ds] < need[ds] - 20}
assert not missing, f"Extraction not finished yet, videos still missing: {missing}. Wait for notebook 02, then re-run this cell."
results = []
for train_on in ["ffpp", "celebdf_strict"]:
    for seed in [0, 1, 2]:
        results.append(T.run(dict(name="v1_mean", train_on=train_on, head="mean", seed=seed)))
print("PHASE 2 RUNS COMPLETE")
'''),
code('''# Cell 5 - Phase 2 summary: mean +- std over seeds, saved to Drive
tab = T.collect()
tab = tab[tab.name == "v1_mean"]
summary = tab.groupby(["train_on", "test"]).agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
          bal_acc=("bal_acc", "mean"), real_rec=("real_rec", "mean"), fake_rec=("fake_rec", "mean"),
          seeds=("seed", "nunique")).round(4)
print(summary.to_string())
summary.to_csv(paths.REPORTS / "phase2_v1_summary.csv")
pm = {}
for f in sorted(paths.RUNS.glob("v1_mean_ffpp_s*/metrics.json")):
    pm[f.parent.name] = json.loads(f.read_text())["ffpp"]["per_method_auc"]
print("\\nFF++ per-method AUC (FF++-trained):"); print(pd.DataFrame(pm).round(4).to_string())
'''),
]
