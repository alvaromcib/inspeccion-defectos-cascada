"""Convert NEU-DET (VOC XML) to YOLO format with a stratified 80/20 split.

Usage: python src/prepare_neu.py <NEU-DET folder with IMAGES/ and ANNOTATIONS/> datasets/neu
"""
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

C = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def main(src: Path, out: Path):
    imgs = {p.stem: p for p in src.rglob("*.jpg") if "Validation" not in str(p) and "Workflow" not in str(p)}
    xmls = {p.stem: p for p in (src / "ANNOTATIONS").glob("*.xml")}
    keys = sorted(set(imgs) & set(xmls))
    random.seed(0)
    by = {}
    for k in keys:
        by.setdefault(k.rsplit("_", 1)[0], []).append(k)
    split = {}
    for ks in by.values():
        random.shuffle(ks)
        n = int(len(ks) * 0.8)
        split.update({k: "train" for k in ks[:n]}); split.update({k: "val" for k in ks[n:]})
    for k in keys:
        root = ET.parse(xmls[k]).getroot()
        W, H = float(root.find("size/width").text), float(root.find("size/height").text)
        lines = []
        for o in root.findall("object"):
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.find(z).text) for z in ("xmin", "ymin", "xmax", "ymax"))
            lines.append(f"{C.index(o.find('name').text.strip())} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        s = split[k]
        (out / "images" / s).mkdir(parents=True, exist_ok=True); (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        shutil.copy(imgs[k], out / "images" / s / f"{k}.jpg")
        (out / "labels" / s / f"{k}.txt").write_text("\n".join(lines))
    print(len(keys), "images ->", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
