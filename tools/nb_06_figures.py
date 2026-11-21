from make_nb import md, code

HEADER = md("""# DeepTrace 06 - report figures from Drive (ROC, confusion matrices, face crops, Grad-CAM)
CPU runtime is enough (GPU optional). Run all. Output: `MyDrive/DeepTrace/figures/` + `figures.zip`.""")

CELLS = [
code('''# Cell 2 - setup
import subprocess, sys, json, tarfile, io, os, numpy as np, pandas as pd, cv2, torch, matplotlib.pyplot as plt
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "facenet-pytorch"], check=True)
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from deeptrace import paths
FIG = paths.DRIVE / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.axisbelow": True, "axes.titleweight": "bold", "legend.frameon": False})
COL = {"ffpp": "#2a78d6", "celebdf_strict": "#eb6834", "celebdf_official": "#1baf7a"}
NAME = {"ffpp": "FF++ test (same dataset)", "celebdf_strict": "Celeb-DF strict (unseen)", "celebdf_official": "Celeb-DF official (unseen)"}
def preds(run, ts): return pd.read_csv(paths.RUNS / run / f"preds_{ts}.csv")
def thr(run): return json.loads((paths.RUNS / run / "metrics.json").read_text())["val"]["threshold"]
'''),
code('''# Cell 3 - ROC curves: final model on all test sets, and v1 vs robust on the unseen dataset
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ts in ["ffpp", "celebdf_strict", "celebdf_official"]:
    d = preds("robust_ffpp_s0", ts); f, t, _ = roc_curve(d.label, d.p)
    axs[0].plot(f, t, color=COL[ts], lw=2, label=f"{NAME[ts]}  AUC {roc_auc_score(d.label, d.p):.3f}")
axs[0].set_title("Final model (robust_ffpp_s0): ROC curves")
for run, c, lab in [("v1_mean_ffpp_s0", "#2a78d6", "v1 per-frame"), ("v2_bilstm_ffpp_s0", "#eb6834", "v2a BiLSTM"),
                    ("v3_freq_ffpp_s0", "#eda100", "v3 + FFT"), ("robust_ffpp_s0", "#e87ba4", "robust aug")]:
    d = preds(run, "celebdf_strict"); f, t, _ = roc_curve(d.label, d.p)
    axs[1].plot(f, t, color=c, lw=2, label=f"{lab}  AUC {roc_auc_score(d.label, d.p):.3f}")
axs[1].set_title("Trained on FF++, tested on unseen Celeb-DF strict")
for a in axs:
    a.plot([0, 1], [0, 1], ":", color="#52514e"); a.set_xlabel("false positive rate (real flagged as fake)")
    a.set_ylabel("true positive rate (fakes caught)"); a.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "fig11_roc_curves.png", dpi=200); plt.show()
'''),
code('''# Cell 4 - confusion matrices of the final model at its validation threshold
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
th = thr("robust_ffpp_s0")
for a, ts in zip(axs, ["ffpp", "celebdf_strict", "celebdf_official"]):
    d = preds("robust_ffpp_s0", ts); cm = confusion_matrix(d.label, d.p >= th, labels=[0, 1])
    a.imshow(cm, cmap="Blues"); a.grid(False)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]}\\n({cm[i, j] / cm[i].sum():.0%})", ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    a.set_xticks([0, 1]); a.set_xticklabels(["pred REAL", "pred FAKE"]); a.set_yticks([0, 1]); a.set_yticklabels(["REAL", "FAKE"])
    a.set_title(NAME[ts], fontsize=10)
fig.suptitle(f"Final model confusion matrices (threshold {th:.3f} chosen on FF++ validation)", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "fig12_confusion_matrices.png", dpi=200); plt.show()
'''),
code('''# Cell 5 - pull a few test videos' crops straight out of the Drive shards (no full restore)
def shard_of(ds):
    m = {}
    for f in sorted(paths.SHARDS.glob(f"status_{ds}_*.json")):
        sh = paths.SHARDS / (f.stem.replace("status_", "") + ".tar")
        for rel, v in json.loads(f.read_text()).items():
            if v["status"] == "ok": m[rel] = sh
    return m
def load_crops(ds, rel, frames=(0, 16, 31), smap={}):
    smap.setdefault(ds, shard_of(ds))
    base = str(Path(rel).with_suffix(""))
    out = {}
    with tarfile.open(smap[ds][rel]) as t:
        for i in frames:
            out[i] = cv2.cvtColor(cv2.imdecode(np.frombuffer(t.extractfile(f"{base}/t0_{i:02d}.jpg").read(), np.uint8), 1), cv2.COLOR_BGR2RGB)
    return out
ff = pd.read_csv(paths.MANIFESTS / "split_ffpp.csv"); ff = ff[(ff.split == "test")]
ff = ff[ff.rel.isin(shard_of("ffpp"))]
src = sorted(set(ff[ff.method == "original"].id_a))[0]                       # one source person, all versions
rows = [("original", ff[(ff.method == "original") & (ff.id_a == src)].rel.iloc[0])]
for m in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
    rows.append((m, ff[(ff.method == m) & (ff.id_a == src)].rel.iloc[0]))
CR = {m: load_crops("ffpp", rel) for m, rel in rows}
fig, axs = plt.subplots(3, 5, figsize=(13, 8))
for j, (m, _) in enumerate(rows):
    for i, fr in enumerate([0, 16, 31]):
        axs[i, j].imshow(CR[m][fr]); axs[i, j].axis("off")
        if i == 0: axs[i, j].set_title(m + (" (real)" if m == "original" else " (fake)"), fontsize=10)
fig.suptitle(f"Face crops of FF++ test video {src}: the original and its four manipulations (frames 0/16/31)", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "fig13_face_crops_ffpp.png", dpi=200); plt.show()
'''),
code('''# Cell 6 - Grad-CAM of the final model on those faces (frame 16)
from deeptrace.models.net import Net
from deeptrace.inference import Detector
info = json.loads((paths.DRIVE / "app_weights" / "model_info.json").read_text())
det = Detector.__new__(Detector); det.dev = "cpu"
det.model = Net(info["head"], info["freq"], pretrained=False); det.model.load_state_dict(torch.load(paths.DRIVE / "app_weights" / "model.pt", map_location="cpu")); det.model.eval()
def frame_score(img):
    x = torch.from_numpy(img).permute(2, 0, 1)[None, None]
    with torch.no_grad():
        return torch.sigmoid(det.model.fc(det.model.frame_features(x)[0])).item()
fig, axs = plt.subplots(2, 5, figsize=(13, 5.6))
for j, (m, _) in enumerate(rows):
    img = CR[m][16]
    axs[0, j].imshow(img); axs[0, j].set_title(m, fontsize=10)
    axs[1, j].imshow(det.gradcam(img)); axs[1, j].set_title(f"fake score {frame_score(img):.2f}", fontsize=9)
    axs[0, j].axis("off"); axs[1, j].axis("off")
fig.suptitle("Grad-CAM of the final model (red = regions that pushed the score towards FAKE)", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "fig14_gradcam_examples.png", dpi=200); plt.show()
'''),
code('''# Cell 7 - Celeb-DF examples (unseen dataset): one real, two fakes, with Grad-CAM
cs = pd.read_csv(paths.MANIFESTS / "split_celebdf_strict.csv"); cs = cs[cs.split == "test"]
cs = cs[cs.rel.isin(shard_of("celebdf"))]                                   # only videos with crops
pick = [("REAL", cs[cs.label == 0].rel.iloc[0]), ("FAKE", cs[cs.label == 1].rel.iloc[0]), ("FAKE", cs[cs.label == 1].rel.iloc[1])]
fig, axs = plt.subplots(2, 3, figsize=(9, 6))
for j, (lab, rel) in enumerate(pick):
    img = load_crops("celebdf", rel, frames=(16,))[16]
    axs[0, j].imshow(img); axs[0, j].set_title(f"{lab}: {Path(rel).name}", fontsize=8)
    axs[1, j].imshow(det.gradcam(img)); axs[1, j].set_title(f"fake score {frame_score(img):.2f}", fontsize=9)
    axs[0, j].axis("off"); axs[1, j].axis("off")
fig.suptitle("Celeb-DF v2 (never seen in training) with Grad-CAM", fontweight="bold")
plt.tight_layout(); plt.savefig(FIG / "fig15_gradcam_celebdf.png", dpi=200); plt.show()
'''),
code('''# Cell 8 - copy the result CSVs next to the figures and zip everything for download
import shutil
for f in ["all_runs.csv", "degradation_all.csv", "mouth_eval_all.csv", "bootstrap_vs_v1.csv", "timing.csv", "extraction_status.csv"]:
    if (paths.REPORTS / f).exists(): shutil.copy(paths.REPORTS / f, FIG / f)
shutil.make_archive(str(paths.DRIVE / "figures"), "zip", FIG)
print("Download:", paths.DRIVE / "figures.zip"); print(sorted(p.name for p in FIG.iterdir()))
'''),
]
