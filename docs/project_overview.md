# DeepTrace: Project Overview (blueprint v0.1, 21 Nov 2026)

## 1. Title
DeepTrace: A Research-Driven, Cloud-Trained Video Deepfake Detector with Controlled Tests of Mouth Dependence, Temporal Modelling, Frequency Cues, Multi-Face Handling and Compression Robustness.

## 2. Summary
DeepTrace detects face-manipulated videos. It samples frames, finds every face, and scores face tracks with an EfficientNet-B0 backbone plus optional temporal and frequency branches. It returns a video verdict, a frame-level timeline and Grad-CAM heatmaps. The research aim is not a higher Celeb-DF score, which is already saturated (~0.99 AUC in-dataset). The aim is controlled, multi-seed evidence on the five reviewed papers' limitations, measured **across datasets** (FaceForensics++ c23 ↔ Celeb-DF v2) and under **real video re-encoding**.

## 3. Background and motivation
Face-swap and reenactment fakes are cheap to make. Published detectors report near-perfect in-dataset scores but drop sharply on unseen datasets and compressed video. The five reviewed papers each leave a stated gap (see §9).

## 4. Problem statement
Given a video, decide whether any face in it is manipulated, report a calibrated-as-possible score and the frames that drove the decision, and measure how that decision holds up on unseen datasets, unseen manipulation methods, missing mouth evidence and real compression.

## 5. Objectives
O1 clean, reproducible pipeline · O2 cross-dataset evaluation · O3 test each gap G1–G5 with a controlled experiment · O4 explainability · O5 a deployed web app · O6 full documentation.

## 6. Existing work summary
Two earlier pipelines exist (see `initial_project_audit.md`):
- **Pipeline A** (TF, frozen EfficientNet-B0 + BiLSTM): official-test acc 0.826 / AUC 0.902. Verified from saved notebook outputs.
- **Pipeline B** (PyTorch, fine-tuned EfficientNet-B0): official AUC ~0.999, strict-split AUC ~0.995 over 3 seeds. **Unverified**: no logs survive.

## 7. Audit findings (headline)
Validation in Pipeline A was not identity-disjoint; Pipeline B's artifacts were lost to non-persistent storage; label conventions differ between the two; only one dataset was used; multi-face, frequency, mouth, codec and cross-dataset tests were never run.

## 8. Papers
LipForensics (CVPR 2021), FTCN (ICCV 2021), MVIM (TDSC 2025), Yan et al. (CVPR 2025), Celeb-DF (CVPR 2020). See `literature_review.md`. Full PDF review is pending upload.

## 9. Research gaps
G1 mouth dependence · G2 temporal benefit · G3 multi-face / localization · G4 generalization / single-artifact reliance · G5 compression · G6 protocol · G7 efficiency. See `research_gap_analysis.md`.

## 10. Datasets

| Dataset | Content | Role |
|---|---|---|
| Celeb-DF v2 | 890 real, 5,639 fake, one face-swap method | Train / test; official and strict splits |
| FF++ c23 | 1,000 real + 4–6 manipulation methods (counts to be verified) | Train / test; LOMO; cross-dataset |

## 11. System architecture

```
video ─► validate/metadata ─► sample 32 frames ─► MTCNN (all faces) ─► IoU tracks
      ─► per track: 16 crops 224×224 ─► EfficientNet-B0 (fine-tuned) ─► [T,1280]
            ├─ head: mean | BiLSTM-attn | temporal-diff conv
            └─ (v3) FFT branch ─► fused
      ─► track score ─► video score = max over tracks ─► locked threshold
      ─► JSON + frame timeline + Grad-CAM
```

## 12. ML methodology
v1 per-frame baseline → v2 temporal heads → v3 frequency fusion → robustness augmentation. Every step is compared under the protocol in `experiment_protocol.md` (3 seeds, bootstrap CIs, cross-dataset primary).

## 13. Cloud
Colab (T4) + Google Drive persistence; Kaggle as the data source; HF Spaces + Vercel for deployment. See `cloud_architecture.md`.

## 14. Stack
Python 3, PyTorch, torchvision, facenet-pytorch (MTCNN), MediaPipe, OpenCV, ffmpeg, scikit-learn, pytorch-grad-cam; FastAPI, Uvicorn; Next.js; Docker.

## 15. Roadmap
P0 plan → P1 data → P2 v1 → P3 v2/v3/E2/robustness/multi-face → P4 inference + XAI → P5 web → P6 deploy → P7 report. See `implementation_plan.md`.

## 16. Deliverables
Package + notebooks, manifests, run folders with logs and checkpoints, `model_comparison.md`, `inference.py`, web app, Docker files, deployed URLs, all docs, final report.

## 17. Risks
Free-Colab GPU availability and disconnects; Drive space; FF++ Kaggle mirror contents and licence unverified; papers not yet uploaded; Celeb-DF saturation hides differences; synthetic multi-face test is weaker than real labels.

## 18. Success criteria
1. Every reported number regenerates from saved files.
2. Every gap G1–G5 has a run experiment with a CI-backed conclusion (positive or negative).
3. Cross-dataset AUC reported in both directions.
4. Web app processes an uploaded video end-to-end on the deployed URLs, verified in a browser.

## Status of results
| Kind | Status |
|---|---|
| Verified existing results | Pipeline A test metrics only |
| Proposed methodology | This document |
| Expected outcomes | In-dataset AUC ≥0.98; cross-dataset well below that (literature typically ~0.65–0.80 FF++→CDF for plain CNNs). **Expectations, not results** |
| Actual new results | None yet |
