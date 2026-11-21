# DeepTrace: Deepfake Video Detection

A research-driven video deepfake detector. It is an EfficientNet-B0 spatio-temporal model trained on **FaceForensics++ (c23)** and evaluated **across datasets** on **Celeb-DF v2**, with controlled experiments on the limitations of five published papers. A Streamlit web app gives a verdict, a frame-by-frame score timeline and Grad-CAM heatmaps.

> The prediction is automated and is not a guarantee of authenticity or manipulation.

## Live demo
Streamlit Community Cloud: *(link added after deployment)*

## What it does
1. **Validates** the upload and reads 32 frames (every 10th, ~10–13 s).
2. **Detects and tracks every face** (MTCNN + IoU tracking, up to 3 faces).
3. **Scores each face** with EfficientNet-B0 (fine-tuned) on 16 frames; the most suspicious face decides the verdict.
4. **Explains** the decision with a per-frame score timeline and Grad-CAM heatmaps on the most suspicious frames.
5. **Exports** a JSON report.

## Repository layout
```
app.py                    Streamlit front end (upload -> verdict -> timeline -> heatmaps -> report)
deeptrace/inference.py    standalone inference (CLI: python -m deeptrace.inference video.mp4)
deeptrace/data/           inventory, leakage-safe splits, face extraction, clip dataset
deeptrace/models/net.py   EfficientNet-B0 + mean / BiLSTM / temporal-difference heads + FFT branch
deeptrace/train.py        resumable training, validation-only thresholds, multi-test evaluation
deeptrace/experiments.py  degradation, mouth-occlusion (E2) and timing experiments
notebooks/                Colab notebooks that ran every experiment
docs/                     audit, plan, protocol, gap analysis, walkthroughs, results
weights/                  model.pt + model_info.json used by the app
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Research summary
See `docs/` (results tables in `docs/results.md`). Main protocol points:
- identity-disjoint splits (0 shared identities)
- thresholds chosen on validation only
- AUC as the primary metric
- cross-dataset tests in both directions
