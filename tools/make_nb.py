"""Build a Colab notebook that installs the deeptrace package onto Drive from an
embedded tarball, so Colab never depends on hand-edited notebook code."""
import base64, io, json, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pkg_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for p in sorted((ROOT / "deeptrace").rglob("*.py")):
            t.add(p, arcname=str(p.relative_to(ROOT)))
    return base64.b64encode(buf.getvalue()).decode()


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


BOOT = '''# Cell 1 - mount Drive and install the deeptrace package from this notebook's embedded copy
from google.colab import drive
drive.mount("/content/drive")
import base64, io, tarfile, sys, os, hashlib
PKG_B64 = "{b64}"
CODE = "/content/drive/MyDrive/DeepTrace/code"
os.makedirs(CODE, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PKG_B64)), mode="r:gz") as t:
    t.extractall(CODE)
sys.path.insert(0, CODE)
print("package version", hashlib.md5(PKG_B64.encode()).hexdigest()[:10], "->", CODE)
'''


def build(cells, out):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "cells": cells}
    Path(out).write_text(json.dumps(nb, indent=1))
    print("wrote", out, Path(out).stat().st_size, "bytes")


if __name__ == "__main__":
    name = sys.argv[1]
    outname = sys.argv[2] if len(sys.argv) > 2 else name
    spec = __import__(f"nb_{name}")
    build([spec.HEADER, code(BOOT.format(b64=pkg_b64()))] + spec.CELLS, ROOT / "notebooks" / f"{outname}.ipynb")
