"""Assemble docs/ from the latest cascade results (VLM run if present, else detector-only preview)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
res_dir = ROOT / "results"
# Priority: VLM run with the current fusion rule > raw VLM run > detector-only preview.
fused, raw = res_dir / "cascade_ollama_fused.json", res_dir / "cascade_ollama.json"
if fused.exists() and (not raw.exists() or fused.stat().st_mtime >= raw.stat().st_mtime):
    src = fused                                       # same run, re-fused with the current rule
elif raw.exists():
    src = raw
else:
    src = res_dir / "cascade_detector.json"
data = json.loads(src.read_text(encoding="utf-8"))
(res_dir / "results.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
img = ROOT / "docs" / "img"
img.mkdir(parents=True, exist_ok=True)
for p in img.glob("crop_*.jpg"):                     # drop crops from earlier runs
    p.unlink()
for e in data["examples"]:
    shutil.copy(res_dir / e["img"], img / e["img"])
strip = Image.open(res_dir / "strip_overview.jpg").rotate(90, expand=True)   # band reads left -> right
strip.thumbnail((4000, 300))
strip.save(img / "strip_overview_h.jpg", quality=80)
subprocess.run([sys.executable, str(ROOT / "shared" / "build.py"), str(ROOT)], check=True)
print("built from", src.name)
