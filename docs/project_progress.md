# Project Progress

| Phase | Status | Notes |
|---|---|---|
| 0 Audit + plan | **Approved** 21 Nov 2026 | 7 docs in `docs/` |
| 1 Data pipeline | **In progress** | Inventory and splits verified; pilot ok; full extraction running (21 Nov, ~19:55 IST) |
| 2 Model v1 | Code ready | `train.py`, `models/net.py`, `evaluate.py`; notebook 03; smoke test pending |
| 3 v2/v3, E2, LOMO, robustness, multi-face | Code partly ready | Heads (bilstm, tdiff) and freq branch written; mouth views in dataset |
| 4-7 | Not started | |

## Blocked / waiting
- Five paper PDFs and the gap-report PDF (needed for `literature_review.md`)
- FF++ licence confirmation (your side)

## Log
- 21 Nov: Phase 0 approved. Colab Pro account, T4. Inventory done (13,529 videos, 0 unreadable, 0 duplicates).
- 21 Nov: Splits built (FF++ 720/140/140, 0 leakage; CDF strict seed 4, 0 leakage). DeepFakeDetection excluded (no real counterparts).
- 21 Nov: Pilot extraction bug (object-dtype MTCNN output) found and fixed; retries added. Full extraction started.
