"""Central paths. Persistent artifacts live on Google Drive; raw videos and
working crops live on the ephemeral Colab disk and are re-created per session."""
import os
from pathlib import Path

DRIVE = Path(os.environ.get("DT_DRIVE", "/content/drive/MyDrive/DeepTrace"))
LOCAL = Path(os.environ.get("DT_LOCAL", "/content/dt_local"))

MANIFESTS = DRIVE / "manifests"     # inventory, profiles, splits (CSV)
SHARDS    = DRIVE / "crops"         # tar shards of face crops (persistent)
RUNS      = DRIVE / "runs"          # one folder per training run
REPORTS   = DRIVE / "reports"       # data-quality reports, figures

CROPS     = LOCAL / "crops"         # unpacked crops for fast training I/O

for p in (MANIFESTS, SHARDS, RUNS, REPORTS, CROPS):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:          # read-only / no Drive (e.g. the deployed app): paths unused there
        pass

FAKE, REAL = 1, 0                   # one label convention everywhere
