"""Assemble docs/ from the latest cascade results (VLM run if present, else detector-only preview)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
res_dir = ROOT / "results"
src = res_dir / "cascade_ollama.json"
if not src.exists():
    src = res_dir / "cascade_detector.json"
shutil.copy(src, res_dir / "results.json")
img = ROOT / "docs" / "img"
img.mkdir(parents=True, exist_ok=True)
for p in res_dir.glob("crop_*.jpg"):
    shutil.copy(p, img / p.name)
strip = Image.open(res_dir / "strip_overview.jpg").rotate(90, expand=True)   # band reads left -> right
strip.thumbnail((4000, 300))
strip.save(img / "strip_overview_h.jpg", quality=80)
subprocess.run([sys.executable, str(ROOT / "shared" / "build.py"), str(ROOT)], check=True)
print("built from", src.name)
