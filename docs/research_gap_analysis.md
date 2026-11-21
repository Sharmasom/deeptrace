# Research-Gap Analysis

**Source caveat:** the five paper PDFs and the gap report PDF were not uploaded in this session. Gaps below come from the summaries in your project notes. Upload the PDFs and I will check every row against the papers and correct this document.

Contribution type: **R** reproduction · **E** engineering improvement · **C** research contribution (controlled finding) · **N** potentially novel (none claimed).

---

## Gap list

### G1. Mouth dependence (LipForensics, CVPR 2021)
1. **Limitation:** a mouth-motion detector; fails when the mouth is occluded, unaltered or barely moving; needs large lip-reading pretraining.
2. **Paper:** LipForensics (stated in its own limitations).
3. **Why it matters:** many real-world clips have masks, hands, microphones or silent subjects. Face-swap fakes often alter the whole face, not just the mouth.
4. **Existing work:** E2 code written, never run. **Not addressed.**
5. **Solution:** full-face model with no visual-speech pretraining. Test how much it depends on the mouth.
6. **Implementation:** MediaPipe lip box per video; three input views (full / mouth-masked / mouth-only); mouth-motion terciles.
7. **Experiment E2:** train on each view, evaluate 3×3 train-view × test-view grid on both datasets; AUC per motion tercile; per FF++ method (NeuralTextures edits only the mouth region, so it is the critical case).
8. **Metrics:** AUC, AUC drop under masking, AUC by motion tercile, with bootstrap CIs.
9. **Conclusion:** see the traceability table and results.md. Supported only if full-face AUC under masking stays high and does not drop in the low-motion tercile. **Type: C.**

### G2. Temporal modelling claim (FTCN, ICCV 2021)
1. **Limitation:** heavy fully-temporal design; claims temporal coherence aids generalization.
2. **Paper:** FTCN.
3. **Why:** if a cheap per-frame model generalizes equally, the temporal cost is not justified.
4. **Existing work:** in-dataset only. cnn ≈ lstm ≈ bilstm on Celeb-DF (UNV numbers). **Partly.**
5. **Solution:** compare per-frame vs temporal heads **cross-dataset**, where FTCN says temporal matters.
6. **Implementation:** heads = mean (v1), BiLSTM-attention (v2a), temporal-difference 1D-conv head (v2b, lightweight FTCN-inspired, *not* FTCN).
7. **Experiment E1′:** train on FF++, test on Celeb-DF and the reverse; 3 seeds; parameters, latency.
8. **Metrics:** cross-dataset AUC (main), in-dataset AUC, ms/video, params.
9. **Conclusion:** see the traceability table and results.md. **Type: C** (a negative result is reportable).

### G3. Multi-face and localization (MVIM, TDSC 2025)
1. **Limitation:** multi-face videos and localizing which face or frames are fake.
2. **Paper:** MVIM.
3. **Why:** largest-face-only pipelines miss a fake face next to a real one.
4. **Existing work:** none.
5. **Solution:** detect and track all faces; score each track; video score = max over tracks; frame-level score timeline; Grad-CAM heatmaps.
6. **Implementation:** MTCNN all faces + IoU tracking; per-track inference.
7. **Experiment E6′:** neither dataset has multi-face labels, so build a **synthetic** test: side-by-side composites (real + fake, real + real) from test videos. Measure largest-face vs all-faces scoring. Grad-CAM pointing-game against FF++ masks only if the Kaggle copy contains masks.
8. **Metrics:** AUC on composites, track-level accuracy.
9. **Conclusion:** see the traceability table and results.md. Synthetic evidence only; say so. **Type: E.**

### G4. Generalization and single-artifact reliance (Yan et al., CVPR 2025)
1. **Limitation:** detectors overfit one artifact type and fail on unseen methods and datasets; cost.
2. **Paper:** Yan et al. CVPR 2025.
3. **Why:** deployed detectors meet unseen generators.
4. **Existing work:** none (one dataset only). **Barely.**
5. **Solution:** (a) frequency branch (DCT/FFT) fused with RGB features; (b) cross-dataset and leave-one-manipulation-out (LOMO) protocols.
6. **Implementation:** log-magnitude FFT of the grey face → small CNN → concatenated with EfficientNet features (v3).
7. **Experiments E3 + E4:** v1 vs v3 on FF++→CDF, CDF→FF++, LOMO on FF++ (train 3 methods, test the 4th); 3 seeds.
8. **Metrics:** cross-dataset AUC, LOMO AUC per held-out method.
9. **Conclusion:** see the traceability table and results.md. **Type: C.**

### G5. Compression and resolution (Celeb-DF, CVPR 2020)
1. **Limitation:** detection degrades on compressed or low-quality video.
2. **Paper:** Celeb-DF.
3. **Why:** social media re-encodes everything.
4. **Existing work:** crop-level JPEG/downscale only (UNV). **Partly.**
5. **Solution:** compression- and resolution-aware augmentation; evaluation by **real video re-encode**.
6. **Implementation:** ffmpeg H.264 at CRF 23/30/35/40 and 0.5× / 0.25× resolution on test videos, then the full pipeline (detection included); train-time augmentation adds random downscale and JPEG.
7. **Experiment E5′:** v_best with vs without robustness augmentation.
8. **Metrics:** AUC per condition, drop from clean, real-class recall at the locked threshold.
9. **Conclusion:** see the traceability table and results.md. **Type: E/C.**

### G6. Evaluation-protocol weaknesses (cross-cutting)
Identity leakage, test-set thresholds, accuracy under imbalance, single seeds. Fixed by the experiment protocol (identity-disjoint splits, validation-only thresholds, AUC + balanced accuracy, 3 seeds, bootstrap CIs). **Type: E.**

### G7. Efficiency (E8)
Measured params, ms/video (GPU and CPU), peak memory for every model. **Type: E.**

---

## Traceability table

| Paper | Limitation | Gap | Solution | Implementation | Experiment | Result | Conclusion |
|---|---|---|---|---|---|---|---|
| LipForensics | Mouth occluded / static | G1 | Full-face, mouth-ablation | views + MTCNN mouth box | E2 | Low-motion AUC 0.958 vs high 0.975; mask: face swap −0.03/−0.06, NT 0.946→0.622; masked training 0.909 | Full face covers low motion and face swaps; mouth is essential for mouth reenactment |
| FTCN | Temporal cost vs benefit | G2 | Per-frame vs temporal heads cross-dataset | v1, v2a, v2b | E1′ | CDF strict 0.869 / 0.852 / 0.862, n.s. | No benefit from temporal heads here |
| MVIM | Multi-face, localization | G3 | All-face tracking, timeline, Grad-CAM | tracker + per-track scoring | not run (no labels) | — | Implemented, not evaluated (future work) |
| Yan CVPR 2025 | Single-artifact reliance, generalization | G4 | Frequency branch; cross-dataset, LOMO | v3, LOMO | E3, E4 | FF++→CDF 0.868; CDF→FF++ 0.664; NT unseen 0.783; FFT −0.016 (sig.) | Gaps confirmed; FFT branch negative |
| Celeb-DF | Compression | G5 | Robust augmentation; degradation test | aug + crop-level JPEG/downscale | E5′ | Cross JPEG30 0.777→0.842; CDF official +0.022 (sig.) | Robustness improved; crop-level only |
| (all) | Protocol | G6 | Strict protocol | splits, locked thresholds | all | 0 identity leakage; val-only thresholds; bootstrap CIs | Done |
| (all) | Cost | G7 | Measure | timing script | E8 | 4.0M params, 14 ms GPU, 0.94 s CPU per clip | Done |

Rule: a gap is "addressed" only when its experiment shows a measured effect with a CI. Implementing a technique is not enough.
