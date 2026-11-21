import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(14, 4.6)); ax.set_xlim(0, 14); ax.set_ylim(0, 4.6); ax.axis("off")
steps = [("Video", "mp4 / mov / avi\nvalidated", "#e8f0fb"),
         ("Frame sampling", "32 frames\n(every 10th,\n~10-13 s)", "#e8f0fb"),
         ("Face detection", "MTCNN, all faces\n+ IoU tracking\n(up to 3 faces)", "#e8f0fb"),
         ("Face crops", "224 x 224 px\n1.3x margin\n16 frames per clip", "#e8f0fb"),
         ("EfficientNet-B0", "ImageNet init,\nblocks 5+ fine-tuned\n-> 1280-d per frame", "#fde9df"),
         ("Temporal head", "mean (v1, final)\nBiLSTM (v2a)\ntemporal-diff (v2b)\n+FFT branch (v3)", "#fde9df"),
         ("Decision", "fake score vs\nvalidation threshold\nmost suspicious face", "#e2f5ec"),
         ("Explain", "frame timeline\nGrad-CAM heatmaps\nJSON report", "#e2f5ec")]
w, gap, y = 1.5, 0.22, 1.2
for i, (t, d, c) in enumerate(steps):
    x = 0.15 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y), w, 2.4, boxstyle="round,pad=0.02,rounding_size=0.12", fc=c, ec="#52514e", lw=1))
    ax.text(x + w / 2, y + 2.05, t, ha="center", va="center", fontsize=10, fontweight="bold", color="#0b0b0b")
    ax.text(x + w / 2, y + 1.0, d, ha="center", va="center", fontsize=8, color="#52514e")
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + w + gap - 0.02, y + 1.2), xytext=(x + w + 0.02, y + 1.2), arrowprops=dict(arrowstyle="->", color="#0b0b0b", lw=1.4))
ax.text(7, 4.25, "DeepTrace pipeline (training and inference use identical preprocessing)", ha="center", fontsize=13, fontweight="bold")
for x0, x1, lab, c in [(0.15, 6.95, "data pipeline (deeptrace/data/prep.py)", "#2a78d6"), (7.03, 10.4, "model (deeptrace/models/net.py)", "#eb6834"),
                       (10.47, 13.85, "output (inference.py + app.py)", "#1baf7a")]:
    ax.plot([x0, x1], [0.85, 0.85], color=c, lw=3); ax.text((x0 + x1) / 2, 0.55, lab, ha="center", fontsize=9, color=c)
ax.text(0.15, 0.1, "Trained on Google Colab (2 x T4, Drive-backed, resumable) · deployed as a Streamlit web app · code: github.com/Sharmasom/deeptrace",
        fontsize=8, color="#52514e")
fig.savefig("/mnt/user-data/outputs/figures/fig00_pipeline.png", dpi=200, bbox_inches="tight"); print("ok")
