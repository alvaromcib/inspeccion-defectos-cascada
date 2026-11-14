"""Assemble docs/ (Spanish) and docs/en/ (English) from the latest inspection run and the VLM experiment."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
res_dir = ROOT / "results"
data = json.loads((res_dir / "inspection.json").read_text(encoding="utf-8"))
exp = ROOT / "experiments" / "vlm" / "summary.json"
data["vlm_experiment"] = json.loads(exp.read_text(encoding="utf-8")) if exp.exists() else None
(res_dir / "results.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

img = ROOT / "docs" / "img"
img.mkdir(parents=True, exist_ok=True)
for p in img.glob("crop_*.jpg"):                     # drop crops from earlier runs
    p.unlink()
for e in data["examples"]:
    shutil.copy(res_dir / e["img"], img / e["img"])
strip = Image.open(res_dir / "strip_overview.jpg").rotate(90, expand=True)   # horizontal strip for the page
strip.thumbnail((4000, 300))
strip.save(img / "strip_overview_h.jpg", quality=80)

subprocess.run([sys.executable, str(ROOT / "shared" / "build.py"), str(ROOT)], check=True)
