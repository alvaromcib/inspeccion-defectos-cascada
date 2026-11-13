"""Download NEU-DET (public GitHub mirror) and convert it to YOLO format in datasets/neu."""
import subprocess
from pathlib import Path

from prepare_neu import main as prepare

ROOT = Path(__file__).resolve().parents[1]
raw = ROOT / "datasets" / "NEU-DET-raw"
if not raw.exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection", str(raw)], check=True)
prepare(raw, ROOT / "datasets" / "neu")
