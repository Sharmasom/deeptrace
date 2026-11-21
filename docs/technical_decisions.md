# Technical Decisions

| # | Date | Decision | Reason | Alternatives rejected |
|---|---|---|---|---|
| D1 | 21 Nov | Label convention fake=1 everywhere | Earlier pipelines disagreed (audit) | — |
| D2 | 21 Nov | Exclude FF++ DeepFakeDetection from train/test | Copy lacks its real actor videos, so it creates a scene shortcut | Keep as fake-only extra test (possible later; reports fake recall only) |
| D3 | 21 Nov | FF++ split by connected components of source IDs, 72/14/14 | Official JSON absent; reproduces 720/140/140 with 0 leakage | Random video split (leaks identities) |
| D4 | 21 Nov | Celeb-DF official split: identity-disjoint validation | Fix A3; validation drives checkpoint and threshold | Stratified random val (old notebook) |
| D5 | 21 Nov | Sequential read, every 10th frame, 32 frames | Seeking 1080p H.264 is too slow on Colab CPUs | Uniform seeking over the whole video |
| D6 | 21 Nov | MTCNN keep_all + IoU tracking, up to 3 tracks, landmarks stored | Multi-face gap G3; mouth boxes for E2 | Largest face only |
| D7 | 21 Nov | Threshold = Youden J on validation; AUC primary | Imbalance; F1 threshold collapsed before | F1-max threshold |
| D8 | 21 Nov | Code embedded in each notebook and unpacked to Drive | No GitHub link; reproducible, no hidden notebook state | Manual %%writefile cells |
