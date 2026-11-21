"""Two-worker sprint notebook: build with WORKER=A or B."""
import os
from make_nb import md, code

W = os.environ.get("WORKER", "A")
N_A = 800        # FF++ videos extracted by worker A after Celeb-DF; B takes the rest

HEADER = md(f"""# DeepTrace sprint - WORKER {W}
Runtime **T4 GPU**. Two copies of this notebook run in parallel (worker A and worker B) and share work through Drive.
**Runtime -> Run all.** If it disconnects, Run all again: extraction, training and evaluation all resume.
Worker A: Celeb-DF extraction + first {N_A} FF++ videos, then training queue A + mouth (E2) evaluation + timing.
Worker B: remaining FF++ videos, then training queue B + degradation evaluation.""")

SETUP = f'''# Cell 2 - setup: installs, credentials, datasets, splits
WORKER = "{W}"
import subprocess, sys, os, json, time, torch, pandas as pd
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "facenet-pytorch"], check=True)
from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME"); os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
import kagglehub
from deeptrace import paths
from deeptrace.data import prep
from deeptrace import train as T, experiments as X
prep.TAG = WORKER
inv = pd.read_csv(paths.MANIFESTS / "inventory.csv")
ff_rows = T.load_split("ffpp").sort_values("rel").reset_index(drop=True)          # 5,000 videos, 4 methods
cdf_rows = inv[inv.dataset == "celebdf"].sort_values("rel").reset_index(drop=True)
t0 = time.time()
CDF = next(p.parent for p in Path(kagglehub.dataset_download("reubensuju/celeb-df-v2")).rglob("List_of_testing_videos.txt"))
FF = Path(kagglehub.dataset_download("xdxd003/ff-c23"))
print(f"worker {{WORKER}} | data ready in {{time.time()-t0:.0f}}s | cpus {{os.cpu_count()}} | {{torch.cuda.get_device_name(0)}}")
print("ffpp videos:", len(ff_rows), "| celebdf videos:", len(cdf_rows))
'''

EXTRACT = {
"A": f'''# Cell 3 - extraction share of worker A (resumable)
prep.run(cdf_rows, "celebdf", CDF)
prep.run(ff_rows.iloc[:{N_A}], "ffpp", FF)
''',
"B": f'''# Cell 3 - extraction share of worker B (resumable)
prep.run(ff_rows.iloc[{N_A}:], "ffpp", FF)
'''}[W]

BARRIER = '''# Cell 4 - wait until BOTH workers have finished extracting (pulls the other worker's shards from Drive)
need = {"celebdf": set(cdf_rows.rel), "ffpp": set(ff_rows.rel)}
while True:
    left = {ds: len(need[ds] - prep.restore(ds)) for ds in need}
    print(time.strftime("%H:%M:%S"), "videos still being extracted:", left, flush=True)
    if all(v == 0 for v in left.values()):
        break
    time.sleep(60)
print("ALL CROPS AVAILABLE")
'''

def wait_fn():
    return '''def wait_for(run_id):
    while not (paths.RUNS / run_id / "metrics.json").exists():
        print(time.strftime("%H:%M:%S"), "waiting for", run_id, flush=True); time.sleep(60)
'''

TRAIN = {
"A": '''# Cell 5 - training queue A (each run ~10-13 min on T4; finished runs are skipped)
Q = [dict(name="v1_mean",    train_on="ffpp", head="mean"),
     dict(name="v2_bilstm",  train_on="ffpp", head="bilstm"),
     dict(name="v3_freq",    train_on="ffpp", head="mean", freq=True),
     dict(name="e2_masked",  train_on="ffpp", head="mean", view="mouth_masked"),
     dict(name="v1_mean",    train_on="ffpp", head="mean", seed=1),
     dict(name="v1_mean",    train_on="ffpp", head="mean", seed=2)]
for cfg in Q:
    try:
        T.run(cfg)
    except Exception as e:
        print("RUN FAILED", cfg, repr(e))
print("QUEUE A DONE")
''',
"B": '''# Cell 5 - training queue B (finished runs are skipped)
ROBUST = dict(p_jpeg=0.5, jpeg_min=20, p_blur=0.3, p_down=0.4)
Q = [dict(name="v1_mean",    train_on="celebdf_strict", head="mean"),
     dict(name="v2_tdiff",   train_on="ffpp", head="tdiff"),
     dict(name="robust",     train_on="ffpp", head="mean", aug=ROBUST),
     dict(name="lomo_noNT",  train_on="ffpp", head="mean", methods=["Deepfakes", "Face2Face", "FaceSwap"]),
     dict(name="v2_bilstm",  train_on="celebdf_strict", head="bilstm"),
     dict(name="v1_mean",    train_on="celebdf_strict", head="mean", seed=1),
     dict(name="v1_mean",    train_on="celebdf_strict", head="mean", seed=2)]
for cfg in Q:
    try:
        T.run(cfg)
    except Exception as e:
        print("RUN FAILED", cfg, repr(e))
print("QUEUE B DONE")
'''}[W]

EVAL = {
"A": wait_fn() + '''# Cell 6 - E2 mouth evaluation (G1) and timing (G7)
X.mouth_eval("v1_mean_ffpp_s0", "ffpp")
X.mouth_eval("e2_masked_ffpp_s0", "ffpp")
wait_for("v1_mean_celebdf_strict_s0")
X.mouth_eval("v1_mean_celebdf_strict_s0", "celebdf_strict")
tim = []
for rid in ["v1_mean_ffpp_s0", "v2_bilstm_ffpp_s0", "v3_freq_ffpp_s0"]:
    tim.append(X.timing(rid))
for rid in ["v2_tdiff_ffpp_s0"]:
    wait_for(rid); tim.append(X.timing(rid))
pd.DataFrame(tim).round(2).to_csv(paths.REPORTS / "timing.csv", index=False)
print(pd.DataFrame(tim).round(2).to_string())
print("EVAL A DONE")
''',
"B": wait_fn() + '''# Cell 6 - degradation evaluation (G5), crop level, threshold fixed from clean validation
for rid in ["v1_mean_celebdf_strict_s0", "v1_mean_ffpp_s0", "robust_ffpp_s0", "v2_bilstm_ffpp_s0",
            "v2_tdiff_ffpp_s0", "v3_freq_ffpp_s0"]:
    wait_for(rid)
    if not (paths.RUNS / rid / "degradation.csv").exists():
        X.degradation(rid)
print("EVAL B DONE")
'''}[W]

SUMMARY = '''# Cell 7 - summary of everything finished so far (safe to run any time)
tab = T.collect()
tab = tab[~tab.name.str.startswith("smoke")]
print(tab.groupby(["name", "train_on", "test"]).agg(auc=("auc", "mean"), auc_std=("auc", "std"), n=("seed", "nunique"),
      bal_acc=("bal_acc", "mean"), real_rec=("real_rec", "mean"), fake_rec=("fake_rec", "mean")).round(4).to_string())
tab.to_csv(paths.REPORTS / "all_runs.csv", index=False)
pm = {f.parent.name: json.loads(f.read_text())["ffpp"].get("per_method_auc", {}) for f in sorted(paths.RUNS.glob("*_ffpp_s*/metrics.json"))}
print("\\nFF++ test AUC per method:"); print(pd.DataFrame(pm).T.round(4).to_string())
deg = [pd.read_csv(f) for f in paths.RUNS.glob("*/degradation.csv")]
if deg:
    deg = pd.concat(deg); deg.to_csv(paths.REPORTS / "degradation_all.csv", index=False)
    print("\\nDegradation AUC:"); print(deg.pivot_table(index=["run", "test"], columns="cond", values="auc").round(4).to_string())
mo = [pd.read_csv(f) for f in paths.RUNS.glob("*/mouth_eval_*.csv")]
if mo:
    mo = pd.concat(mo); mo.to_csv(paths.REPORTS / "mouth_eval_all.csv", index=False)
    print("\\nMouth evaluation:"); print(mo.round(4).to_string())
'''

CELLS = [code(SETUP), code(EXTRACT), code(BARRIER), code(TRAIN), code(EVAL), code(SUMMARY)]
