# Literature Review

**Status: incomplete.** The five paper PDFs were not uploaded in this session. The table below uses your earlier gap report summary only. Every cell marked "to verify" will be filled from the PDFs once you upload them, before Phase 3 design is frozen.

| Paper | Objective / method (summary) | Datasets | Stated limitations | To verify from PDF |
|---|---|---|---|---|
| LipForensics, Haliassos et al., CVPR 2021 | Detects unnatural mouth motion with a lip-reading-pretrained spatio-temporal network | FF++, CDF, DFDC, DFo | Fails with occluded / static / unaltered mouth; needs large lip-reading pretraining | metrics, cross-dataset numbers, exact limitation wording |
| FTCN, Zheng et al., ICCV 2021 | Fully temporal convolution (spatial kernel 1) + temporal transformer, for temporal incoherence | FF++, CDF, DFDC, DFo | Heavier design; claims temporal modelling aids generalization | architecture size, cross-dataset AUC |
| MVIM, TDSC 2025 | Multi-view inconsistency modelling with localization for multi-face videos | to verify | Multi-face localization scope | full details |
| Yan et al., CVPR 2025 | Video-level blending data + spatiotemporal adapter for generalizable video detection | to verify | Generalization, cost, over-reliance on one artifact type | full details |
| Celeb-DF, Li et al., CVPR 2020 | Higher-quality face-swap benchmark; evaluates many detectors | CDF v1/v2 | Hard benchmark; compression reduces performance | protocol details |
