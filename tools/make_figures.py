"""Report/PPT figures from the saved DeepTrace results (numbers copied from metrics.json via the
Colab summary; learning curves parsed from the executed notebooks)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path("/mnt/user-data/outputs/figures")
S = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
C = {"v1": "#2a78d6", "bilstm": "#eb6834", "tdiff": "#1baf7a", "freq": "#eda100", "robust": "#e87ba4",
     "masked": "#008300", "lomo": "#6250d6", "cdf": "#e34948"}
NAMES = {"v1": "v1 per-frame (mean)", "bilstm": "v2a BiLSTM", "tdiff": "v2b temporal-diff", "freq": "v3 + FFT branch",
         "robust": "robust augmentation", "masked": "mouth-masked training", "lomo": "trained without NT"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": S["ink2"], "axes.labelcolor": S["ink"],
    "xtick.color": S["ink2"], "ytick.color": S["ink2"], "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": S["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False, "axes.titleweight": "bold",
    "axes.titlesize": 12})

def save(fig, name, src):
    fig.text(0.01, 0.005, src, fontsize=7, color=S["ink2"], ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / name, dpi=200); plt.close(fig); print("saved", name)

def bars(ax, groups, series, values, colors, labels, err=None, ylim=None, fmt="{:.3f}"):
    n = len(series); w = 0.8 / n; x = np.arange(len(groups))
    for i, s in enumerate(series):
        v = values[s]
        b = ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.9, color=colors[s], label=labels[s],
                   yerr=None if err is None else err.get(s), capsize=3, error_kw=dict(ecolor=S["ink2"], lw=1))
        for xi, vi in zip(x + (i - (n - 1) / 2) * w, v):
            if not np.isnan(vi):
                ax.text(xi, vi + (ylim[1] - ylim[0]) * 0.01 if ylim else vi, fmt.format(vi), ha="center", va="bottom",
                        fontsize=7, color=S["ink2"], rotation=90 if n > 3 else 0)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    if ylim: ax.set_ylim(*ylim)

SRC = "DeepTrace · identity-disjoint test sets · threshold from validation only"

# 1. dataset splits
sp = pd.DataFrame({"split": ["FF++ train", "FF++ val", "FF++ test", "CDF strict train", "CDF strict val", "CDF strict test",
                             "CDF official test"],
                   "real": [720, 140, 140, 552, 119, 219, 178], "fake": [2880, 560, 560, 2051, 101, 500, 340]})
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(sp)); ax.bar(x - 0.2, sp.real, 0.38, color=C["v1"], label="real"); ax.bar(x + 0.2, sp.fake, 0.38, color=C["bilstm"], label="fake")
for xi, r, f in zip(x, sp.real, sp.fake):
    ax.text(xi - 0.2, r + 30, r, ha="center", fontsize=7, color=S["ink2"]); ax.text(xi + 0.2, f + 30, f, ha="center", fontsize=7, color=S["ink2"])
ax.set_xticks(x); ax.set_xticklabels(sp.split, rotation=20, ha="right"); ax.set_ylabel("videos")
ax.set_title("Dataset splits (0 identities shared between train / val / test)"); ax.legend()
save(fig, "fig01_dataset_splits.png", "FF++ c23: 4 methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures), split by source-video pairs · Celeb-DF v2 strict split by identity (seed 4)")

# 2. cross-dataset generalization (v1, 3 seeds)
fig, ax = plt.subplots(figsize=(8, 4.2))
lab = ["FF++ → FF++\n(same)", "FF++ → CDF strict\n(unseen)", "FF++ → CDF official\n(unseen)", "CDF → CDF strict\n(same)", "CDF → FF++\n(unseen)"]
m = [0.9716, 0.8676, 0.8152, 0.9957, 0.6642]; sd = [0.0022, 0.0015, 0.0081, 0.0017, 0.0095]
col = [C["v1"]] * 3 + [C["cdf"]] * 2
ax.bar(range(5), m, 0.6, yerr=sd, capsize=4, color=col, error_kw=dict(ecolor=S["ink"], lw=1))
for i, (a, s) in enumerate(zip(m, sd)): ax.text(i, a + 0.012, f"{a:.3f} ± {s:.3f}", ha="center", fontsize=8, color=S["ink"])
ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=8); ax.set_ylim(0.5, 1.03); ax.set_ylabel("ROC-AUC")
ax.axhline(0.5, color=S["ink2"], lw=0.8, ls=":"); ax.set_title("Cross-dataset generalization gap (Model v1, mean ± std over 3 seeds)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C["v1"], label="trained on FF++"), Patch(color=C["cdf"], label="trained on Celeb-DF")], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
save(fig, "fig02_cross_dataset.png", SRC)

# 3. model comparison
ms = ["v1", "bilstm", "tdiff", "freq", "robust"]
vals = {"v1": [0.9737, 0.8689, 0.8218], "bilstm": [0.9708, 0.8522, 0.8104], "tdiff": [0.9691, 0.8621, 0.8083],
        "freq": [0.9634, 0.8528, 0.8124], "robust": [0.9606, 0.8811, 0.8441]}
fig, ax = plt.subplots(figsize=(9, 4.4))
bars(ax, ["FF++ test (same dataset)", "Celeb-DF strict (unseen)", "Celeb-DF official (unseen)"], ms, vals, C, NAMES, ylim=(0.75, 1.01))
ax.set_ylabel("ROC-AUC"); ax.set_title("Model variants trained on FF++ (seed 0)"); ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
save(fig, "fig03_model_comparison.png", SRC + " · all models EfficientNet-B0 backbone, same data and schedule")

# 4. bootstrap CIs vs v1
ci = pd.DataFrame([["robust", "FF++", -0.0132, -0.0206, -0.0066], ["robust", "CDF strict", 0.0122, -0.0048, 0.0297], ["robust", "CDF official", 0.0223, 0.0024, 0.0434],
    ["bilstm", "FF++", -0.0027, -0.0092, 0.0039], ["bilstm", "CDF strict", -0.0167, -0.0342, 0.0000], ["bilstm", "CDF official", -0.0108, -0.0303, 0.0102],
    ["tdiff", "FF++", -0.0045, -0.0121, 0.0027], ["tdiff", "CDF strict", -0.0067, -0.0242, 0.0114], ["tdiff", "CDF official", -0.0130, -0.0387, 0.0127],
    ["freq", "FF++", -0.0104, -0.0166, -0.0043], ["freq", "CDF strict", -0.0162, -0.0343, -0.0001], ["freq", "CDF official", -0.0088, -0.0284, 0.0106]],
    columns=["m", "test", "d", "lo", "hi"])
fig, ax = plt.subplots(figsize=(8, 5))
y = np.arange(len(ci))[::-1]
for yi, r in zip(y, ci.itertuples()):
    sig = not (r.lo <= 0 <= r.hi)
    ax.plot([r.lo, r.hi], [yi, yi], color=C[r.m], lw=2)
    ax.plot(r.d, yi, "o", ms=8, color=C[r.m] if sig else "white", mec=C[r.m], mew=2)
ax.axvline(0, color=S["ink"], lw=1)
ax.set_yticks(y); ax.set_yticklabels([f"{NAMES[r.m]} · {r.test}" for r in ci.itertuples()], fontsize=8)
ax.set_xlabel("AUC difference vs v1 (95% paired bootstrap CI)")
ax.set_title("Which changes really matter? (filled = significant)")
save(fig, "fig04_bootstrap_ci.png", SRC + " · 1,000 paired bootstrap resamples on the same test videos")

# 5. per-method AUC incl. LOMO
meth = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
pm = {"v1": [0.9819, 0.9742, 0.9870, 0.9435], "robust": [0.9726, 0.9725, 0.9915, 0.9057],
      "masked": [0.9571, 0.8939, 0.9535, 0.8296], "lomo": [0.9770, 0.9736, 0.9904, 0.7829]}
fig, ax = plt.subplots(figsize=(9, 4.4))
bars(ax, meth, ["v1", "robust", "masked", "lomo"], pm, C, {**NAMES, "v1": "v1 (mean of 3 seeds)"}, ylim=(0.7, 1.04))
ax.set_ylabel("ROC-AUC (method fakes vs all real)"); ax.set_title("FF++ test AUC per manipulation method")
ax.annotate("unseen in training:\n0.944 → 0.783", xy=(3.3, 0.79), xytext=(2.15, 0.715), fontsize=8, color=S["ink"],
            arrowprops=dict(arrowstyle="->", color=S["ink2"]))
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
save(fig, "fig05_per_method_auc.png", SRC + " · LOMO = leave NeuralTextures out of training (CVPR 2025 gap)")

# 6. mouth ablation (LipForensics)
views = {"full face": [0.984, 0.9757, 0.9887, 0.9463], "mouth masked": [0.8255, 0.7049, 0.9242, 0.6221], "mouth only": [0.8302, 0.855, 0.8257, 0.7588]}
vc = {"full face": C["v1"], "mouth masked": C["bilstm"], "mouth only": C["tdiff"]}
fig, ax = plt.subplots(figsize=(9, 4.4))
bars(ax, meth, list(views), views, vc, {k: f"tested on {k}" for k in views}, ylim=(0.5, 1.03))
ax.axhline(0.5, color=S["ink2"], lw=0.8, ls=":")
ax.set_ylabel("ROC-AUC"); ax.set_title("How much does the full-face model rely on the mouth? (FF++ test)"); ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
save(fig, "fig06_mouth_ablation.png", "Full-face v1 model (seed 0) evaluated with a grey box over the MTCNN mouth region, or on the mouth crop only (LipForensics gap)")

# 7. mouth-motion terciles
fig, ax = plt.subplots(figsize=(7.5, 4.2))
g = ["low", "mid", "high"]
lines = {"FF++, full face": ([0.9584, 0.9673, 0.975], C["v1"], "-"), "FF++, mouth masked": ([0.706, 0.7356, 0.8067], C["bilstm"], "-"),
         "FF++, masked-trained on masked": ([0.8661, 0.9013, 0.9242], C["masked"], "-"), "Celeb-DF strict, full face": ([0.9984, 0.99, 0.9961], C["cdf"], "--")}
for k, (v, c, ls) in lines.items():
    ax.plot(g, v, ls, color=c, lw=2, marker="o", ms=8, mec="white", mew=2, label=k)
    ax.text(2.05, v[-1], f"{v[-1]:.3f}", va="center", fontsize=8, color=S["ink2"])
ax.set_ylim(0.65, 1.01); ax.set_xlabel("mouth-motion tercile (n ≈ 233–240 videos each)"); ax.set_ylabel("ROC-AUC")
ax.set_title("Low mouth motion: full-face detection stays accurate"); ax.legend(fontsize=8, loc="lower right")
save(fig, "fig07_mouth_motion.png", "Motion = std over time of mouth width / eye distance from MTCNN landmarks")

# 8. degradation
conds = ["clean", "jpeg50", "jpeg30", "down112", "down64", "jpeg50+down112"]
deg = {"CDF strict (unseen)": {"v1": [0.8689, 0.827, 0.7765, 0.841, 0.7985, 0.7833], "robust": [0.8811, 0.8644, 0.8417, 0.8607, 0.827, 0.8324],
                              "bilstm": [0.8522, 0.815, 0.7534, 0.8237, 0.7159, 0.7533], "tdiff": [0.8621, 0.82, 0.7597, 0.8188, 0.7244, 0.76],
                              "freq": [0.8527, 0.8241, 0.7929, 0.8246, 0.7743, 0.7807]},
       "FF++ (same dataset)": {"v1": [0.9737, 0.9431, 0.9067, 0.9285, 0.8104, 0.8888], "robust": [0.9606, 0.9321, 0.918, 0.9229, 0.8609, 0.8961],
                              "bilstm": [0.9708, 0.9434, 0.9034, 0.9188, 0.8038, 0.8909], "tdiff": [0.9691, 0.9325, 0.8963, 0.9076, 0.7895, 0.8684],
                              "freq": [0.9634, 0.9277, 0.887, 0.9021, 0.7723, 0.8616]}}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, (t, d) in zip(axs, deg.items()):
    for m_ in ["v1", "bilstm", "tdiff", "freq", "robust"]:
        ax.plot(conds, d[m_], color=C[m_], lw=3 if m_ in ("v1", "robust") else 1.5, marker="o", ms=7 if m_ in ("v1", "robust") else 5,
                mec="white", mew=1.5, label=NAMES[m_], alpha=1 if m_ in ("v1", "robust") else 0.8)
    ax.set_title(t); ax.tick_params(axis="x", rotation=25); ax.set_ylim(0.68, 1.0)
axs[0].set_ylabel("ROC-AUC"); axs[1].legend(fontsize=8, loc="lower left")
fig.suptitle("Robustness to compression and low resolution (FF++-trained models)", fontweight="bold")
save(fig, "fig08_degradation.png", "Crop-level degradation of test faces (JPEG quality / downscale to N px and back), threshold fixed from clean validation")

# 9. learning curves
h = pd.read_csv("/tmp/history.csv")
key = {"v1_mean_ffpp_s0": "v1", "v2_bilstm_ffpp_s0": "bilstm", "v2_tdiff_ffpp_s0": "tdiff", "v3_freq_ffpp_s0": "freq", "robust_ffpp_s0": "robust",
       "e2_masked_ffpp_s0": "masked", "lomo_noNT_ffpp_s0": "lomo"}
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
for r, k in key.items():
    d = h[h.run == r]
    axs[0].plot(d.epoch, d.val_auc, color=C[k], lw=2, marker="o", ms=5, mec="white", label=NAMES[k])
    axs[1].plot(d.epoch, d.loss, color=C[k], lw=2, marker="o", ms=5, mec="white", label=NAMES[k])
for r, ls, mk in [("v1_mean_celebdf_strict_s0", "-", "o"), ("v1_mean_celebdf_strict_s1", "--", "s"), ("v1_mean_celebdf_strict_s2", ":", "^"), ("v2_bilstm_celebdf_strict_s0", "-", "D")]:
    d = h[h.run == r]; k = "bilstm" if "bilstm" in r else "cdf"
    axs[2].plot(d.epoch, d.val_auc, ls, color=C[k], lw=2, marker=mk, ms=6, mec="white",
                label=("BiLSTM, seed 0" if "bilstm" in r else "v1, seed " + r[-1]))
axs[0].set_title("FF++ training: validation AUC"); axs[1].set_title("FF++ training: training loss"); axs[2].set_title("Celeb-DF strict: validation AUC")
for a in axs: a.set_xlabel("epoch")
axs[0].set_ylabel("val ROC-AUC"); axs[1].set_ylabel("weighted BCE loss"); axs[0].legend(fontsize=7); axs[2].legend(fontsize=7)
save(fig, "fig09_learning_curves.png", "8 epochs, AdamW + cosine schedule, mixed precision, Colab T4 · best-validation checkpoint kept")

# 10. efficiency
eff = pd.DataFrame({"m": ["v1", "bilstm", "tdiff", "freq"], "params": [4.01, 7.16, 5.06, 4.12], "gpu": [14.27, 15.85, 15.82, 16.77]})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, col, t, u in [(axs[0], "params", "Parameters", "million"), (axs[1], "gpu", "GPU time per 16-frame clip (T4)", "ms")]:
    ax.barh([NAMES[m_] for m_ in eff.m], eff[col], color=[C[m_] for m_ in eff.m], height=0.6)
    for i, v in enumerate(eff[col]): ax.text(v, i, f" {v:.2f}", va="center", fontsize=8, color=S["ink"])
    ax.set_title(t); ax.set_xlabel(u); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
save(fig, "fig10_efficiency.png", "Temporal heads add 1–3 M parameters without improving AUC · CPU ≈ 0.9–1.1 s per clip")
