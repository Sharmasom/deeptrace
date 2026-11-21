# Experiment Protocol

These rules are fixed before any test result is seen. Changing them later means recording the change and the reason in `technical_decisions.md`.

## Data and splits
- **Labels:** fake = 1, real = 0, everywhere.
- **Celeb-DF:** (a) official 518-video test list (benchmark comparability) and (b) a strict identity-disjoint split (identities assigned to train/val/test; mixed-identity fakes dropped; seed recorded). Validation is always identity-disjoint from training.
- **FF++ c23:** split by **source video ID**. If the official 720/140/140 split JSONs exist they are used; otherwise a seeded grouped split where `original k` and every fake that uses `k` as target or source fall in the same split. A fake whose two IDs fall in different splits is dropped.
- Frames of one video never cross splits; manifests are saved as CSV with hashes.

## Training
- Same preprocessing for all models: 32 frames sampled, 16 used; MTCNN; 1.3× margin; 224×224.
- Optimiser, schedule, epochs and batch fixed per experiment family and written in `config.yaml`.
- Checkpoint selection: best **validation AUC**. The test set is never used for selection.

## Thresholds
- Threshold = Youden J on validation. Locked (hash) before test evaluation.
- Threshold-free AUC is the primary metric.

## Metrics reported for every run
ROC-AUC (primary), PR-AUC, balanced accuracy, accuracy, precision / recall / F1 per class, confusion matrix, all at the locked threshold. Video-level, plus frame-level AUC where relevant.

## Statistics
- 3 seeds for every comparison used in a claim; mean ± std.
- Paired bootstrap (1,000 resamples) 95% CI on AUC differences on the same test set.
- A difference is called a ranking only if the CI excludes 0. Otherwise: "no measurable difference".

## Cross-dataset
- FF++→CDF: train/val on FF++, test on the CDF official list and the strict CDF test. CDF→FF++: reverse, tested on the FF++ test split.
- Thresholds always come from the source dataset's validation set.

## Robustness
- Real re-encode with ffmpeg (libx264 CRF 23/30/35/40; scale 0.5 and 0.25) of **test videos only**, then the full pipeline from the decoded video.
- Crop-level JPEG/downscale kept as a secondary, cheaper check.

## Pilot first
Every experiment family first runs a ≤5-minute pilot (≈100 videos, 1 epoch) to catch errors before GPU time is spent.

## Logging
Every run writes `config.yaml`, `history.csv`, `val_preds.csv`, `test_preds.csv`, `metrics.json` and the checkpoint to Drive. Report tables are generated from these files by code.
