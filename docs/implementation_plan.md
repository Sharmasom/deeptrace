# Implementation Plan

Each phase ends with a walkthrough in `docs/walkthroughs/` and stops for your approval.

## Repository layout (to be created in Phase 1)

```
deeptrace/
  configs/            yaml per experiment
  deeptrace/
    data/             inventory.py, splits.py, prep.py (frames, MTCNN, multi-face tracks), dataset.py, augment.py
    models/           backbone.py, heads.py (mean, bilstm_attn, tdiff_conv), freq.py, net.py
    train.py          AMP, resume, early stopping, logging to Drive
    evaluate.py       metrics, locked thresholds, bootstrap CIs
    robustness.py     ffmpeg re-encode + crop-level degradations
    mouth.py          lip boxes, views, motion terciles (E2)
    explain.py        Grad-CAM, frame timeline
  notebooks/
    00_setup_colab.ipynb      mount Drive, Kaggle download, install
    01_prep.ipynb             preprocessing (resumable)
    02_train.ipynb            runs from a run queue (resumable)
    03_eval.ipynb             tables and plots from saved predictions
  inference.py
  app/backend (FastAPI), app/frontend (Next.js)
  docker/, docs/, experiments/
```

Code lives in your `C:\deepfake` folder and is synced to Drive. Colab notebooks only call the package, so no logic sits in notebook state.

## Task list with dependencies

| ID | Task | Depends on | GPU |
|---|---|---|---|
| P0 | Phase 0 docs (this) + your approval | — | no |
| P1.1 | Colab setup: Drive mount, Kaggle Secrets (you add the key), download both datasets | P0 | no |
| P1.2 | Inventory + profiling (counts, fps, resolution, duration, duplicates by hash) | P1.1 | no |
| P1.3 | Splits: CDF official + strict; FF++ grouped by source ID | P1.2 | no |
| P1.4 | Pilot extraction (100 videos) + visual check | P1.3 | yes |
| P1.5 | Full extraction, all faces with tracks, sharded to Drive | P1.4 | yes, ~4 h |
| P2 | Model v1 (EffNet-B0 mean): FF++ and CDF, 3 seeds, in- and cross-dataset | P1.5 | ~1 h |
| P3.1 | v2a BiLSTM-attn, v2b temporal-difference conv (G2) | P2 | ~2 h |
| P3.2 | v3 RGB + FFT branch (G4) + LOMO on FF++ | P2 | ~2 h |
| P3.3 | E2 mouth views on both datasets (G1) | P2 | ~1 h |
| P3.4 | Robust augmentation + ffmpeg re-encode eval (G5) | P2 | ~3 h |
| P3.5 | Multi-face synthetic test (G3), E8 timing | P3.1 | ~0.5 h |
| P3.6 | `model_comparison.md`, choose final model by the predefined rule | P3.1–P3.5 | no |
| P4 | `inference.py` + Grad-CAM + timeline | P3.6 | no |
| P5 | FastAPI backend + Next.js frontend | P4 | no |
| P6 | Docker, HF Spaces + Vercel deploy, end-to-end browser test | P5 | no |
| P7 | `final_project_report.md` | P6 | no |

## Final-model selection rule (fixed now)
1. Highest mean **cross-dataset AUC** (average of both directions).
2. Ties (CI overlapping) broken by robustness (mean AUC over re-encode conditions).
3. Then by CPU latency.

## How I will run Colab
I will drive Colab through the browser on your laptop: open the notebooks from Drive, select the T4 runtime, run cells, watch logs and restart from checkpoints when a session drops. Things only you can do: sign in to Google if asked, add the Kaggle token to Colab Secrets, and approve any paid upgrade. Colab sessions stop if the tab is closed or the laptop sleeps, so the laptop must stay awake during runs.
