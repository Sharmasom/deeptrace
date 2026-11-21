# Initial Project Audit (Phase 0)

Date: 21 Nov 2026. Author: Claude, for Amritansh Vivek.
Status legend: **VC** verified correct · **VI** verified incorrect · **PC** partially correct · **INC** incomplete · **UNV** unverified.

"Verified" here means checked against code or saved notebook outputs I could read. Nothing was re-executed: I had no GPU, no dataset and no checkpoints when writing this.

---

## 1. Inventory of materials received

| Item | Received? | Notes |
|---|---|---|
| `updated_work.ipynb` (TensorFlow/Keras, Kaggle) | Yes | Has saved outputs: training logs, validation and official-test metrics |
| `DeepTrace_Project_Notes.md` (PyTorch pipeline + results) | Yes | Code is pasted in the notes; results are numbers you typed back. No logs, CSVs or checkpoints |
| Master project prompt | Yes | Governs this workflow |
| Five research papers (PDF) | **No** | Only summaries in the project notes |
| Research-gap document (`DeepTrace_Research_Gap_and_Modification_Report.pdf`) | **No** | Only its summary in the project notes |
| Checkpoints, CSV logs, screenshots | **No** | The notes say crops, `meta.csv` and checkpoints were wiped from `/kaggle/working` |
| `C:\deepfake` folder on the laptop | Empty | No local code to audit |
| Kaggle datasets | Links only | Celeb-DF verified indirectly through notebook outputs; FF++ c23 not inspected yet |

---

## 2. Two separate previous pipelines

There are **two different codebases** with very different results. The rebuild has to explain the gap between them, not just pick the better number.

| | Pipeline A: `updated_work.ipynb` | Pipeline B: project notes |
|---|---|---|
| Framework | TensorFlow 2.20 / Keras | PyTorch |
| Face detection | Haar / RetinaFace / RetinaFace + 5-point alignment | MTCNN, largest face, 1.3x margin |
| Frames | 20 per video | 32 saved, 16 used |
| Backbone | EfficientNet-B0 **frozen** (feature cache) | EfficientNet-B0, blocks 5+ **fine-tuned** |
| Head | Stacked BiLSTM(128) → BiLSTM(64) → Dense | mean (cnn) / LSTM / BiLSTM + attention pooling |
| Labels | Fake = 0, **Real = 1** | Real = 0, **Fake = 1** |
| Split | Official 518 test; train/val **stratified random** | Official 518 test + strict identity-disjoint split |
| Best official-test result | retina_raw: acc 0.826, AUC 0.902 | bilstm: AUC 0.9997 (seed 0) |

**Key unresolved question:** AUC ~0.90 (frozen) vs ~0.999 (fine-tuned) on the same 518 test videos. Fine-tuning plus augmentation can plausibly explain much of it, but Pipeline B's numbers have no surviving logs, so this gap is **UNV** until the rebuild reproduces it.

---

## 3. Pipeline A audit (`updated_work.ipynb`)

| # | Finding | Status |
|---|---|---|
| A1 | Dataset counts asserted: 590 Celeb-real, 300 YouTube-real, 5,639 fake; official test 518 = 340 fake + 178 real | **VC** (assertions passed in saved output) |
| A2 | Official test list reserved before splitting; asserted equal to test set | **VC** |
| A3 | Train/val split is `train_test_split(..., stratify=label)`: **not grouped by identity**. Same celebrities appear in train and val, and val drives early stopping and threshold choice, so validation scores are optimistic | **VI** for any generalization claim |
| A4 | Official test identities overlap training (49 shared, measured in Pipeline B) | **VI** as a test of unseen identities; it is still the standard benchmark protocol |
| A5 | Labels Real=1; class weights `balanced` (0: 0.567, 1: 4.22) | **VC** |
| A6 | Thresholds chosen on validation macro-F1, locked with SHA-256 before the test | **VC** (good practice) |
| A7 | Frozen EfficientNet-B0 with ImageNet weights, inputs in [0,255] as Keras expects | **VC**, but it caps performance: the backbone never learns forgery cues |
| A8 | Early stopping on `val_loss`, patience 3; `val_loss` swings strongly (e.g. 0.30 → 0.40 → 0.24 → 0.46), so haar and aligned stopped at epoch 10 | **PC**: training is unstable and the stopping point is noise-driven |
| A9 | Test results match saved outputs: haar acc 0.817 / AUC 0.893; retina_raw 0.826 / 0.902; retina_aligned 0.768 / 0.789 | **VC** |
| A10 | Bootstrap 95% CI, retina_raw − haar accuracy: [−0.019, +0.039], includes 0 | **VC**: no significant difference between Haar and RetinaFace |
| A11 | Alignment is significantly worse (CI [−0.079, −0.015]) | **VC** result. Cause **UNV**; one hypothesis is that the tight 0.85-scaled template crops out the face boundary where blending artifacts sit |
| A12 | Real recall only 0.45–0.62: the model misses ~40% of real videos | **VC**: main weakness |
| A13 | Historical "84.17% with threshold 0.80" from the earlier PDF | **UNV**: it is not shown how 0.80 was chosen; if it was chosen on test, the number is optimistic |
| A14 | Features cached with video hash + model fingerprint; deterministic ops enabled | **VC** (good reproducibility) |
| A15 | No augmentation, no compression study, no cross-dataset test, single seed | **INC** |

---

## 4. Pipeline B audit (PyTorch code in the project notes)

| # | Finding | Status |
|---|---|---|
| B1 | Tensor flow: `B,T,3,224,224` uint8 → normalise → `B·T,1280` → `B,T,1280` → mean or (Bi)LSTM + attention → logit `B` | **VC** (read through the code) |
| B2 | Early backbone blocks frozen with BatchNorm forced to eval in `train()` | **VC** |
| B3 | `pos_weight = n_real / n_fake` with fake = 1: down-weights the majority class | **VC** |
| B4 | Strict split: identities assigned to roles, mixed-identity fakes dropped (2,987), YouTube-real assigned at random | **PC**: correct idea. Loses 53% of fakes, and YouTube-real may share people with Celeb identities, which is untestable without labels |
| B5 | Checkpoint chosen by val AUC; test evaluated once | **VC** |
| B6 | `best_threshold` (F1 over 0.05–0.95) collapses to 0.05 because val is mostly fake; Youden J used afterwards | **PC**: the reported thresholds (0.006–0.03) sit outside the F1 grid, so the two threshold methods are mixed in the results |
| B7 | `cudnn.benchmark=True` plus non-deterministic LSTM: run-to-run noise ~0.0006 | **VC** that noise exists. It is acceptable if reported |
| B8 | Degradation test applied to saved 224×224 crops, not to video re-encodes | **VC** as described. It is a crop-level proxy only |
| B9 | All artifacts in `/kaggle/working`, wiped twice | **VI** (reproducibility failure) |
| B10 | All reported numbers (E1 3-seed, degradation) | **UNV**: no logs or CSVs survive; they come from chat |
| B11 | Only the largest face is kept | **INC** (multi-face gap) |
| B12 | Augmentation has JPEG and blur but no downscaling, which matches the 64×64 weakness | **PC** |
| B13 | E2 mouth code (MediaPipe lips, masks, motion terciles) | **INC**: written, never run |
| B14 | One dataset only | **INC** |

---

## 5. Dataset verification status

| Check | Celeb-DF v2 (`reubensuju/celeb-df-v2`) | FF++ c23 (`xdxd003/ff-c23`) |
|---|---|---|
| Accessible | Yes on Kaggle (used before) | **UNV**: page metadata only; structure not visible without an account |
| Counts | 590 + 300 real, 5,639 fake, 518 official test (**VC**) | **UNV**. Expected: 1,000 originals + 1,000 per method (Deepfakes, Face2Face, FaceSwap, NeuralTextures, possibly FaceShifter and DeepFakeDetection) |
| Identity info | `idX_idY` in filenames; YouTube-real none | Fake names `target_source` (e.g. `000_003`); official 720/140/140 split by source pairs. The split JSONs may not be in the Kaggle copy |
| Duplicates | Not checked | Not checked |
| Resolution / fps | Not profiled | Not profiled |
| Licence | Celeb-DF terms of use (research only) | FF++ requires agreeing to the official terms of use; a Kaggle mirror does not replace that. You should confirm you are covered |

---

## 6. What to keep, what to rebuild

**Reuse (validated reference):** Celeb-DF inventory assertions (A1, A2); validation-locked thresholds with hashes (A6); feature/crop caching keyed by video hash (A14); the PyTorch model and training loop (B1–B3, B5); the strict identity split idea (B4); the E2 mouth code (B13) as a starting point.

**Rebuild or correct:**
1. One label convention everywhere: **fake = 1**.
2. Identity-grouped validation for every split (fixes A3).
3. Persistent storage: crops, manifests, checkpoints and logs on Google Drive, written as they are produced (fixes B9).
4. One threshold rule (Youden J on validation), plus threshold-free AUC (fixes B6).
5. A real ffmpeg re-encode robustness test (extends B8).
6. Downscale augmentation (B12).
7. Multi-face handling (B11).
8. A second dataset and cross-dataset tests (B14).
9. Every number written to CSV in Drive by the code, never copied by hand (fixes B10).

**Retain as baselines in the report:** Pipeline A retina_raw (frozen features; official-test AUC 0.902) as the "frozen-feature" baseline, quoted from its saved outputs.
