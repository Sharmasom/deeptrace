from make_nb import md, code

HEADER = md("""# DeepTrace 01 - Setup, download, inventory (Phase 1.1-1.2)
Runtime: GPU (T4) not needed for this notebook; CPU is fine. Run cells top to bottom.
Kaggle credentials come from **Colab Secrets** `KAGGLE_USERNAME` and `KAGGLE_KEY` (key icon in the left bar, with *Notebook access* switched on).
Outputs are written to `MyDrive/DeepTrace/manifests` and `reports`.""")

CELLS = [
code('''# Cell 2 - environment
import subprocess, sys, shutil, torch, platform
print(platform.python_version(), "torch", torch.__version__, "cuda", torch.cuda.is_available())
print("disk free GB:", round(shutil.disk_usage("/content").free / 1e9, 1))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
'''),
code('''# Cell 3 - Kaggle credentials from Colab Secrets (never printed)
import os
from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
print("kaggle user:", os.environ["KAGGLE_USERNAME"])
'''),
code('''# Cell 4 - download both datasets to the ephemeral disk (~10-20 min)
import kagglehub, time, json
from deeptrace import paths
t0 = time.time()
CDF = kagglehub.dataset_download("reubensuju/celeb-df-v2")
print("Celeb-DF ->", CDF, f"{time.time()-t0:.0f}s")
t0 = time.time()
FF = kagglehub.dataset_download("xdxd003/ff-c23")
print("FF++ c23 ->", FF, f"{time.time()-t0:.0f}s")
(paths.LOCAL / "roots.json").write_text(json.dumps({"celebdf": CDF, "ffpp": FF}))
'''),
code('''# Cell 5 - show the raw layout of both datasets before parsing anything
from deeptrace.data import inventory as inv
inv.tree(CDF, depth=2)
print()
inv.tree(FF, depth=3)
'''),
code('''# Cell 6 - locate dataset roots and parse
from pathlib import Path
cdf_root = next(p.parent for p in Path(CDF).rglob("List_of_testing_videos.txt"))
rows_cdf = inv.celebdf_rows(cdf_root)
rows_ff = inv.ff_rows(FF)
print("celebdf root:", cdf_root, "videos:", len(rows_cdf))
print("ffpp videos parsed:", len(rows_ff))
import collections
print(collections.Counter((r["method"], r["label"]) for r in rows_ff))
'''),
code('''# Cell 7 - profile + hash every video (~15-30 min), save inventory to Drive
import pandas as pd
df = pd.concat([inv.build(rows_cdf, desc="celebdf"), inv.build(rows_ff, desc="ffpp")], ignore_index=True)
test_map = inv.celebdf_official_test(cdf_root)
df["cdf_official_test"] = df.rel.isin(test_map) & (df.dataset == "celebdf")
bad = [r for r, l in test_map.items() if not ((df.rel == r) & (df.label == l)).any()]
print("official test videos matched:", int(df.cdf_official_test.sum()), "| label mismatches:", len(bad))
out = paths.MANIFESTS / "inventory.csv"
df.drop(columns=["path"]).to_csv(out, index=False)
print("saved", out)
inv.summarize(df)
'''),
code('''# Cell 8 - list any FF++ files the parser skipped (unknown folders), so nothing is dropped silently
seen = set(df[df.dataset == "ffpp"].rel)
skipped = [str(p.relative_to(FF)) for p in Path(FF).rglob("*")
           if p.suffix.lower() in inv.VIDEO_EXT and str(p.relative_to(FF)) not in seen]
print("skipped video files:", len(skipped)); print(skipped[:20])
other = sorted({p.suffix for p in Path(FF).rglob("*") if p.is_file() and p.suffix.lower() not in inv.VIDEO_EXT})
print("non-video file types in FF++:", other)
'''),
]
