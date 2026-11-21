from make_nb import md, code

HEADER = md("""# DeepTrace 02 - Splits + face extraction (Phase 1.3-1.5)
Runtime: **T4 GPU**. Run cells in order. Safe to re-run after a disconnect: extraction resumes from the shards saved on Drive.
Needs `manifests/inventory.csv` from notebook 01 and the Kaggle secrets.""")

CELLS = [
code('''# Cell 2 - installs, credentials, datasets (download is skipped if already cached in this runtime)
import subprocess, sys, os, json, time, torch
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "facenet-pytorch"], check=True)
from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME"); os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
import kagglehub
from pathlib import Path
t0 = time.time()
CDF = next(p.parent for p in Path(kagglehub.dataset_download("reubensuju/celeb-df-v2")).rglob("List_of_testing_videos.txt"))
FF = Path(kagglehub.dataset_download("xdxd003/ff-c23"))
print("CDF", CDF, "| FF", FF, f"| {time.time()-t0:.0f}s | cpus {os.cpu_count()} | gpu {torch.cuda.get_device_name(0)}")
'''),
code('''# Cell 3 - build and save splits (leakage-checked)
import pandas as pd
from deeptrace import paths
from deeptrace.data import splits as S
inv = pd.read_csv(paths.MANIFESTS / "inventory.csv")
SP = S.build_all(inv)
for k, d in SP.items():
    d.to_csv(paths.MANIFESTS / f"split_{k}.csv", index=False)
print("\\nsaved to", paths.MANIFESTS)
'''),
code('''# Cell 4 - pilot: 20 videos per dataset, then look at the crops before the full run
from deeptrace.data import prep
ff_rows = SP["ffpp"].sort_values("rel"); cdf_rows = inv[inv.dataset == "celebdf"].sort_values("rel")
pilot_ff = ff_rows.groupby("method", group_keys=False).head(4)
pilot_cdf = cdf_rows.groupby("method", group_keys=False).head(7)
t0 = time.time()
prep.run(pilot_cdf, "celebdf", CDF)
prep.run(pilot_ff, "ffpp", FF)
print(f"pilot {len(pilot_ff)+len(pilot_cdf)} videos in {time.time()-t0:.0f}s")
'''),
code('''# Cell 5 - visual check: frames 0/16/31 of track 0 with landmarks, plus extra tracks if any
import cv2, numpy as np, matplotlib.pyplot as plt
def show(ds, rels, n=6):
    fig, ax = plt.subplots(n, 4, figsize=(9, 2.3 * n))
    for r, rel in enumerate(rels[:n]):
        d = paths.CROPS / ds / Path(rel).with_suffix("")
        if not (d / "meta.json").exists():
            ax[r, 0].set_title(f"NO CROPS {rel[-25:]}", fontsize=7); [a.axis("off") for a in ax[r]]; continue
        m = json.loads((d / "meta.json").read_text())
        for c, i in enumerate([0, 16, 31]):
            a = ax[r, c]; a.axis("off")
            f = d / f"t0_{i:02d}.jpg"
            if f.exists():
                a.imshow(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
                lm = np.array(m["tracks"][0]["landmarks"][i]); a.scatter(lm[:, 0], lm[:, 1], s=6, c="lime")
            a.set_title(f"{Path(rel).parent.name[:12]} f{i}", fontsize=7)
        a = ax[r, 3]; a.axis("off")
        f = d / "t1_16.jpg"
        if f.exists(): a.imshow(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)); a.set_title("track 1", fontsize=7)
    plt.tight_layout(); plt.show()
show("celebdf", list(pilot_cdf.rel.sample(6, random_state=0)))
show("ffpp", list(pilot_ff.rel.sample(6, random_state=0)))
for ds in ["celebdf", "ffpp"]:
    st = prep.status_table(ds); print(ds, st.status.value_counts().to_dict())
    assert not st.status.str.startswith("error").any(), "pilot errors - stop and report"
'''),
code('''# Cell 6 - FULL extraction, Celeb-DF (6,529 videos). Resumable: just re-run after a disconnect.
prep.run(cdf_rows, "celebdf", CDF)
'''),
code('''# Cell 7 - FULL extraction, FF++ without DeepFakeDetection (6,000 videos). Resumable.
prep.run(ff_rows, "ffpp", FF)
'''),
code('''# Cell 8 - data-quality report (saved to Drive)
rep = []
for ds in ["celebdf", "ffpp"]:
    st = prep.status_table(ds).assign(dataset=ds)
    rep.append(st)
rep = pd.concat(rep)
lab = pd.concat([inv[["dataset", "rel", "label", "method"]]])
rep = rep.merge(lab, on=["dataset", "rel"], how="left")
rep.to_csv(paths.REPORTS / "extraction_status.csv", index=False)
print(rep.groupby(["dataset", "status"]).size().to_string())
print("\\ntracks per video:"); print(rep.groupby(["dataset", "n_tracks"]).size().unstack(fill_value=0).to_string())
print("\\nvideos with >1 face in some frame:", rep.groupby("dataset").apply(lambda d: int((d.faces_max > 1).sum())).to_dict())
print("\\nmean detected frames (of 32) by method:"); print(rep.groupby(["dataset", "method"]).detected.mean().round(1).to_string())
'''),
]
