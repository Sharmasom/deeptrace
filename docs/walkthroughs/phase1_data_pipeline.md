# Walkthrough: Phase 1, data pipeline (status: extraction running)

## What was implemented
| File | Purpose |
|---|---|
| `deeptrace/paths.py` | Drive (persistent) vs local (ephemeral) paths; label convention fake=1 |
| `deeptrace/data/inventory.py` | Parse both datasets, profile every video (fps, frames, resolution, duration, size, MD5) |
| `deeptrace/data/splits.py` | FF++ grouped split, Celeb-DF official + strict splits, leakage report |
| `deeptrace/data/prep.py` | Frame sampling, MTCNN (all faces), IoU tracking, crops + landmarks, sharded Drive backup, resume |
| `deeptrace/data/dataset.py` | Clip dataset, clip-consistent augmentation, mouth views for E2 |
| `notebooks/01_setup_inventory`, `02_splits_extract` | Colab drivers; code is embedded and installed to Drive, no logic in notebook state |

## Data flow
```
Kaggle zip -> /content (ephemeral)
 -> inventory.csv (Drive)                  13,529 videos profiled and hashed
 -> split_*.csv (Drive)                    leakage-checked
 -> read sequentially, keep every 10th frame -> 32 frames (~10-13 s window)
 -> MTCNN on frames downscaled to 640 px, keep_all, p >= 0.90
 -> greedy IoU tracks (link IoU > 0.3), keep tracks seen in >= 50% of frames, max 3
 -> gaps filled from nearest detected frame; square crop 1.3x margin -> 224x224 JPEG q95
 -> meta.json: frame indices, timestamps, boxes, 5 landmarks in crop coords
 -> tar shard to Drive every 400 videos + done list (resume point)
Training: track 0, 16 of 32 frames -> uint8 [T,3,224,224]
```

## Verified results
**Inventory (notebook 01):**
- Celeb-DF: 590 Celeb-real + 300 YouTube-real + 5,639 fake = 6,529 videos. All 518 official test videos matched, with 0 label mismatches.
- FF++ c23: 1,000 original plus 1,000 each of Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter and DeepFakeDetection = 7,000 videos.
- Overall: 0 unreadable files and 0 exact duplicates (MD5).
- Celeb-DF is ~30 fps, mostly ~850×480, 12.6 s mean. FF++ is 15–60 fps, 480p to 1080p, 19.3 s mean.

**Splits (notebook 02, Cell 3):**

| Split | Train real/fake | Val | Test | Identity leakage |
|---|---|---|---|---|
| FF++ (grouped by source pairs) | 720 / 3,600 | 140 / 700 | 140 / 700 | 0 / 0 / 0 |
| Celeb-DF official | 609 / 3,595 (1,551 mixed fakes dropped) | 103 / 153 | 178 / 340 | train–val 0; train–test 47 (inherent to the official list) |
| Celeb-DF strict (seed 4) | 552 / 2,051 (2,987 dropped) | 119 / 101 | 219 / 500 | 0 / 0 / 0 |

**Pilot:** 45 videos, all `ok` after the fix below. A visual check confirmed centred crops and correct landmarks across all FF++ methods and Celeb-DF.

## Technical decisions
1. **DeepFakeDetection is excluded.** This copy has its 1,000 fakes but none of the matching real actor videos, so it would teach a scene shortcut (these scenes always mean fake).
2. **FF++ is split by connected components of source IDs.** The official split JSON is absent from this copy. The result reproduces the official 720/140/140 sizes with 0 leakage and drops no videos.
3. **Celeb-DF official split has an identity-disjoint validation set.** This fixes audit finding A3, at the cost of dropping 1,551 mixed fakes from train/val.
4. **Frames are read sequentially (every 10th) instead of uniform seeking.** Seeking in 1080p H.264 was the bottleneck: profiling ran at 3.3 videos/s even with a single seek per video. Spacing is similar to the old pipeline (~0.33–0.4 s).
5. **All faces are tracked (up to 3).** Track 0 = largest face, which equals the old behaviour. The extra tracks serve G3 (multi-face).
6. **Landmarks are stored**, which enables the E2 mouth views without needing MediaPipe.

## Problems encountered
| Problem | Cause | Fix |
|---|---|---|
| 29 of 45 pilot videos errored (`rint` on float object) | MTCNN returns object-dtype arrays when face counts differ across frames | Cast to float64; failed videos are now retried instead of being marked done |
| "Too many sessions" | Notebook 01 runtime still alive | Terminate finished sessions |
| Browser link to the laptop dropped several times | Bridge instability | You run cells; I write code |

## Remaining limitations
- YouTube-real has no identity labels (assigned at random).
- Only the first ~10–13 s of each video is used.
- Gap filling copies the nearest box (no interpolation).
- Multi-face statistics are pending the Cell 8 report.
