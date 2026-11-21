# DeepTrace: Final Project Report

**Cloud-trained video deepfake detection with controlled tests of five published limitations**
Amritansh Vivek · B.Tech CSE, Sikkim Manipal Institute of Technology · November 2026

## 1. Summary
DeepTrace is a deepfake video detector. It:
- tracks every face in a video
- scores each face with a fine-tuned EfficientNet-B0
- explains its verdict with a frame-by-frame score timeline and Grad-CAM heatmaps
- runs as a public Streamlit web app

Training on Celeb-DF alone already gives ~0.996 AUC, so that number is not the contribution. The contribution is a corrected evaluation protocol and seven controlled experiments that test the stated limitations of five papers **across datasets** (FaceForensics++ c23 ↔ Celeb-DF v2). Positive and negative results are reported alike.

## 2. Problem and objectives
Given a video, decide whether a face in it was manipulated, show which frames drove the decision, and measure how far that decision can be trusted on unseen datasets, unseen manipulation methods, missing mouth evidence and degraded quality.

## 3. Starting point (audit)
Two earlier pipelines existed (`initial_project_audit.md`):
- **Pipeline A** (TensorFlow, frozen EfficientNet + BiLSTM): official-test AUC 0.902, real-video recall 45–62%, validation not separated by identity.
- **Pipeline B** (PyTorch, fine-tuned): official AUC ~0.999, but its logs and checkpoints were lost, and it used one dataset only.

The rebuild kept the valid parts and fixed:
- label convention (fake = 1 everywhere)
- identity leakage
- non-persistent storage
- threshold choice (validation only)
- single-dataset evaluation

## 4. Data
| Dataset | Used | Split |
|---|---|---|
| FaceForensics++ c23 (Kaggle mirror) | 1,000 real + Deepfakes, Face2Face, FaceSwap, NeuralTextures (1,000 each) | Grouped by source pairs: 720/140/140 videos, **0 shared identities** |
| Celeb-DF v2 | 890 real, 5,639 fake | Strict identity-disjoint (seed 4, **0 shared**) and official 518-video test |

Verified: 13,529 videos profiled, 0 unreadable, 0 duplicate files.

Excluded: FF++ DeepFakeDetection, because its real actor videos are missing from this copy and it would teach a shortcut. FaceShifter was left out for time.

## 5. Method
```
video -> 32 frames (every 10th) -> MTCNN, all faces -> IoU tracks (<=3) -> 224x224 crops (1.3x margin)
      -> EfficientNet-B0 (blocks 5+ fine-tuned) per frame [T,1280]
      -> head: mean (v1) | BiLSTM+attention (v2a) | temporal-difference conv (v2b) | +FFT branch (v3)
      -> logit; video verdict = most suspicious face vs validation threshold
```

Training setup:
- AdamW (backbone 1e-4, head 1e-3), cosine schedule, 8 epochs, mixed precision
- `pos_weight` for class imbalance
- clip-consistent augmentation; robust variant adds JPEG q20–90 and downscaling to 56–160 px
- Colab Pro on 2 parallel T4 GPUs, with all artifacts on Google Drive and every run resumable

## 6. Results (details in `results.md`)

| Gap (paper) | Experiment | Result | Verdict |
|---|---|---|---|
| Generalization (Yan et al., CVPR 2025) | Cross-dataset, n=3 | FF++ 0.972 → Celeb-DF 0.868 (strict) / 0.815 (official); Celeb-DF 0.996 → FF++ 0.664 | Gap confirmed; asymmetric; training diversity matters |
| Unseen method (Yan et al.) | Leave NeuralTextures out | NT AUC 0.944 → 0.783; other methods unchanged | Cues are method-specific |
| Frequency cues (Yan et al.) | v3 RGB+FFT | −0.010 FF++, −0.016 CDF strict (both significant) | Negative result |
| Temporal modelling (FTCN) | BiLSTM, temporal-difference heads | No significant change; worse at 64 px | Temporal heads add cost, not accuracy, in this lightweight setting |
| Mouth dependence (LipForensics) | Mask / mouth-only, motion terciles | Full-face model works on low-motion videos (0.958 vs 0.975); masking the mouth costs 0.03–0.06 on face swaps but drops NeuralTextures to 0.62; mouth-masked training recovers to 0.909 | Full face helps where LipForensics is weak (low motion, face swaps); mouth evidence is essential for mouth-region reenactment |
| Compression (Celeb-DF) | Robustness augmentation, crop-level degradations | Cross-dataset JPEG 30: 0.777 → 0.842; CDF official +0.022 (sig.); in-dataset −0.013 (sig.) | Robustness improved at a small in-dataset cost |
| Multi-face (MVIM) | All-face tracking in the pipeline and app | Implemented; no multi-face labels in either dataset, so **not quantitatively evaluated** | Engineering only; future work |
| Efficiency | Timing | 4.0 M params, 14 ms/clip GPU, ~0.9 s/clip CPU | Runs on a free CPU host |

**Final model:** `robust_ffpp_s0`, selected by the pre-fixed rule (best mean cross-dataset AUC).

## 7. Contribution statement (what was done beyond each paper)
- **LipForensics:** A full-face detector without lip-reading pretraining stays accurate on low-mouth-motion videos. A controlled masking study shows exactly when mouth evidence is necessary (NeuralTextures, Face2Face) and when it is not (face swaps).
- **FTCN:** A lightweight, controlled comparison across datasets found no generalization benefit from temporal heads on per-frame EfficientNet features. This is not a reimplementation of FTCN.
- **Yan et al. CVPR 2025:** Confirmed the cross-dataset and unseen-method gaps with seeds and confidence intervals; a simple FFT branch did not help; compression-aware augmentation improved cross-dataset AUC on one benchmark.
- **Celeb-DF:** Corrected protocol (identity-disjoint validation, strict split, validation-only thresholds) and a degradation study.
- **MVIM:** Multi-face tracking and per-face scoring implemented in the product; evaluation left to future work.

No novelty is claimed. These are reproductions, engineering improvements and controlled findings.

## 8. System
- **`deeptrace/inference.py`:** standalone; video in, JSON out with label, scores, timestamps, frame scores, faces and Grad-CAM paths.
- **`app.py`:** Streamlit app showing upload and preview, progress, verdict, scores, confidence, frame timeline, suspicious frames, Grad-CAM, a frame inspector, multi-face table, JSON report download and a disclaimer.
- **Deployment:** GitHub → Streamlit Community Cloud (CPU).

## 9. Limitations
1. Single seed for v2/v3/robust/E2/LOMO; only v1 has 3 seeds.
2. Degradation is crop-level, not real video re-encoding.
3. Only the first ~10–13 s of each video is used.
4. YouTube-real videos have no identity labels.
5. The mouth mask is a synthetic grey box.
6. Multi-face is not evaluated; no audio; FaceShifter and DeepFakeDetection are not used.
7. Scores are not calibrated, and thresholds come from the source dataset's validation set.
8. A Grad-CAM heatmap shows model attention, not proof of manipulation.
9. The five paper PDFs were not re-read in this rebuild; the gap list comes from the earlier gap report.

## 10. Future work
Multi-seed runs for every variant, real codec re-encoding, temporal localization, multi-face benchmarks, FaceShifter / DeepFakeDetection / DFDC as unseen sets, probability calibration, and a GPU-backed API.

## 11. Reproducibility
- Every number comes from `runs/*/metrics.json`, `preds_*.csv` and `degradation.csv`, regenerated by notebook cells.
- Splits are in `manifests/`.
- Code is the `deeptrace` package; notebooks only call it.
