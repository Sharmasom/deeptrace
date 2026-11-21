# Cloud Architecture

## Platform choice

| Option | GPU | Persistence | Cost | Verdict |
|---|---|---|---|---|
| **Google Colab (free)** + Google Drive | T4 when available; sessions drop when idle and have a time cap | Drive only (`/content` is wiped) | Free | **Chosen** (your request) |
| Colab Pay-As-You-Go / Pro | T4 / L4 / A100, longer sessions | Drive | Paid compute units (check current prices) | Only if free sessions keep failing; needs your approval |
| Kaggle Notebooks | T4×2 / P100, weekly GPU quota | Committed outputs + Kaggle datasets | Free | Fallback. Both datasets are already attached there |
| RunPod / AWS / GCP | Any | Volumes / buckets | Paid hourly | Not needed for this size |

No paid resource is started without your explicit yes.

## Training data flow (Colab)

```
Kaggle (Celeb-DF v2, FF++ c23)
   │  kagglehub / kaggle CLI using your token from Colab Secrets
   ▼
/content/raw/            (ephemeral, re-downloaded each session, ~10-20 GB per dataset)
   │  prep: frame sampling → MTCNN → 224×224 crops
   ▼
/content/crops/  ──tar shards every 500 videos──►  Drive: DeepTrace/crops/*.tar
   ▼                                                Drive: DeepTrace/manifests/*.csv
train.py (AMP, resume)  ──every epoch──►           Drive: DeepTrace/runs/<run_id>/
                                                        config.yaml, history.csv,
                                                        last.pt, best.pt, preds, metrics.json
```

- **Resume:** preprocessing skips videos already in the shards; training reloads `last.pt` (model, optimiser, scaler, scheduler, epoch, RNG state).
- **Storage budget on Drive:** crops ≈ 0.5 MB per video × ~12,000 videos ≈ 6–7 GB, plus checkpoints (~20 MB each, best + last only) ≈ 1–2 GB. Fits the free 15 GB if your Drive has ~9 GB free. Raw videos are **not** kept on Drive.
- **Secrets:** the Kaggle API token goes into **Colab Secrets** (`KAGGLE_USERNAME`, `KAGGLE_KEY`), which **you** add. I will not type or see the key.
- **Experiment tracking:** CSV/JSON per run in Drive plus a `runs_index.csv`. Weights & Biases is optional and not required.

## Compute estimate (T4)

| Step | Estimate |
|---|---|
| Download both datasets | ~10–20 min per session |
| Face extraction, Celeb-DF (6,529 videos) | ~1.5–2 h |
| Face extraction, FF++ c23 (~5,000–7,000 videos) | ~1.5–2.5 h |
| Model runs (≈45 runs at 7–15 min each) | ~7–9 h |
| Re-encode robustness eval | ~2 h |
| **Total** | **≈13–16 GPU hours** |

On free Colab this means about 4–6 sessions over several days. The plan works in chunks that each finish inside one session.

## Deployment (Phase 6, planned)

| Component | Host | Why |
|---|---|---|
| Model + FastAPI inference API | Hugging Face Spaces (Docker, free CPU) | EfficientNet-B0 at 16 frames runs on CPU in seconds; persistent model file in the Space repo |
| Frontend (Next.js) | Vercel (free) | Static and serverless hosting |
| Job storage | Space-local temp, cleaned after each job | No user video is kept |

GPU inference hosting is not needed at this model size. CPU latency will be measured in E8 before deployment is decided.
