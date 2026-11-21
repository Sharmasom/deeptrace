from make_nb import md, code

HEADER = md("""# DeepTrace 05 - choose final model, export app weights, test inference end-to-end
Run after both sprint notebooks print QUEUE ... DONE. CPU or GPU runtime both work.
Output: `MyDrive/DeepTrace/app_weights/model.pt` + `model_info.json` -> put both in the GitHub repo's `weights/` folder.""")

CELLS = [
code('''# Cell 2 - pick the final model by the rule fixed in the plan:
# highest mean cross-dataset AUC (FF++ -> Celeb-DF strict and official); app needs a full-face model
import json, shutil, subprocess, sys, os, pandas as pd, torch
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "facenet-pytorch", "kagglehub"], check=True)
from deeptrace import paths, train as T
tab = T.collect()
cand = tab[(tab.train_on == "ffpp") & ~tab.name.isin(["e2_masked", "lomo_noNT"]) & ~tab.name.str.startswith("smoke")]
cross = cand[cand.test.isin(["celebdf_strict", "celebdf_official"])].groupby("run").auc.mean()
indom = cand[cand.test == "ffpp"].set_index("run").auc
score = pd.DataFrame({"cross_dataset_auc": cross, "ffpp_test_auc": indom}).sort_values("cross_dataset_auc", ascending=False)
print(score.round(4).to_string())
best = score.index[0]
print("\\nSELECTED:", best)
'''),
code('''# Cell 3 - export weights + model card for the app
rd = paths.RUNS / best
cfg = json.loads((rd / "config.json").read_text()); met = json.loads((rd / "metrics.json").read_text())
out = paths.DRIVE / "app_weights"; out.mkdir(exist_ok=True)
shutil.copy(rd / "best.pt", out / "model.pt")
info = dict(name=best, head=cfg["head"], freq=cfg["freq"], train_on="FaceForensics++ c23 (4 methods)",
            threshold=met["val"]["threshold"],
            test_auc={"FF++ test (same dataset)": met["ffpp"]["auc"],
                      "Celeb-DF strict (unseen dataset)": met["celebdf_strict"]["auc"],
                      "Celeb-DF official (unseen dataset)": met["celebdf_official"]["auc"]})
(out / "model_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2)); print("size MB:", round((out / "model.pt").stat().st_size / 1e6, 1))
'''),
code('''# Cell 4 - end-to-end inference test on real test videos (single files fetched from Kaggle)
from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME"); os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
import kagglehub
from deeptrace.inference import Detector
sp = pd.read_csv(paths.MANIFESTS / "split_celebdf_strict.csv"); te = sp[sp.split == "test"]
picks = list(te[te.label == 0].rel.head(2)) + list(te[te.label == 1].rel.head(2))
det = Detector(out, device="cpu")          # CPU, same as the deployed app
for rel in picks:
    f = kagglehub.dataset_download("reubensuju/celeb-df-v2", path=rel)
    res, vis = det.analyze(f, out_dir=f"/content/infer/{os.path.basename(rel)}")
    print(f"{rel:40s} truth={'FAKE' if 'synthesis' in rel else 'REAL'}  pred={res['predicted_label']}  "
          f"fake_score={res['fake_probability']:.3f}  faces={res['n_faces']}  {res['metadata']['processing_s']}s")
'''),
code('''# Cell 5 - look at one result: timeline + heatmaps
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1 + len(vis["cams"]), figsize=(14, 3))
if res["frame_fake_probability"]:
    ax[0].plot(res["frame_timestamps"], res["frame_fake_probability"]); ax[0].axhline(res["threshold"], ls="--", c="grey")
    ax[0].set_title("frame fake score"); ax[0].set_xlabel("s")
for a, (i, img) in zip(ax[1:], vis["cams"].items()):
    a.imshow(img); a.axis("off"); a.set_title(f"frame {i}")
plt.show()
'''),
]
