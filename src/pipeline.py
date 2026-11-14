"""Automatic inspection of a moving steel strip: detector on every frame + per-section report.

  1. YOLO11n detector on EVERY frame (cheap, ~tens of ms on CPU).
  2. Tracking in strip coordinates (on a real line, the encoder gives the strip position), so each
     physical defect becomes one object with one type, however many frames it is seen in.
  3. Report per strip section with an explicit, configurable decision rule (OK / REVISAR / RECHAZAR).

A local vision-language model (VLM) was tested as a second stage and discarded: it made the defect
type worse (see experiments/vlm/).

Usage:
  python src/pipeline.py --rows 12
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from strip import CLASSES, ES, TILE, build_strip, frames

ROOT = Path(__file__).resolve().parents[1]

# Example decision rule, not learned from data. In a plant, the quality manager sets it per product
# and customer. NEU-DET only contains images with defects, so every section here has findings.
RULE = {
    "critical_types": ["inclusion", "rolled-in_scale"],  # embedded material: harder to fix downstream
    "reject_critical": 3,       # this many critical defects in a section -> RECHAZAR
    "reject_area": 0.5,         # one defect larger than this fraction of a 200x200 px image -> RECHAZAR
    "review_count": 1,          # at least this many defects of any type -> REVISAR
}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def decide(ds, rule=RULE):
    area = [(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) / (TILE * TILE) for d in ds]
    critical = sum(d["label"] in rule["critical_types"] for d in ds)
    if critical >= rule["reject_critical"] or any(a > rule["reject_area"] for a in area):
        return "RECHAZAR"
    return "REVISAR" if len(ds) >= rule["review_count"] else "OK"


def run(weights, data, out_dir, view_h=400, step=8, conf=0.1, rows=30):
    model = YOLO(weights)
    s = build_strip(Path(data), n_rows=rows)
    tracks = defaultdict(lambda: {"boxes": [], "det": [], "conf": [], "last_seen": -99, "ref": (0, 0, 0, 0)})
    n_frames, t_det, timeline = 0, 0.0, []           # timeline: (frame, detections in that frame)
    for fi, y, fr in frames(s, view_h, step):
        n_frames += 1
        t0 = time.time()
        r = model.predict(fr, imgsz=256, conf=conf, verbose=False)[0]
        t_det += time.time() - t0
        for b, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            sb = (b[0], b[1] + y, b[2], b[3] + y)                # box in strip coordinates
            best, best_iou = None, 0.35
            for tid_, tr_ in tracks.items():
                if fi - tr_["last_seen"] > 8:
                    continue
                v = iou(sb, tr_["ref"])
                if v > best_iou:
                    best, best_iou = tid_, v
            tid = best if best is not None else len(tracks)
            tr = tracks[tid]
            tr["boxes"].append(sb); tr["last_seen"] = fi
            tr["ref"] = tuple(np.median(np.array(tr["boxes"][-6:]), axis=0))
            tr["det"].append(CLASSES[c]); tr["conf"].append(cf)
        timeline.append((fi, len(r.boxes)))

    # ---- consolidate tracks: one object per physical defect, type = majority over frames ----
    defects = []
    for tid, tr in tracks.items():
        if len(tr["boxes"]) < 3:                      # flicker: ignore very short tracks
            continue
        bx = np.median(np.array(tr["boxes"]), axis=0).tolist()
        best = max(s.boxes, key=lambda g: iou(bx, g[:4]))
        gt = best[4] if iou(bx, best[:4]) > 0.3 else None
        defects.append({"id": tid, "box": [round(v, 1) for v in bx], "frames": len(tr["boxes"]),
                        "label": Counter(tr["det"]).most_common(1)[0][0], "gt": gt,
                        "mean_conf": round(float(np.mean(tr["conf"])), 3)})
    defects.sort(key=lambda d: -d["frames"])          # merge duplicates of the same physical defect
    merged = []
    for d in defects:
        host = next((h for h in merged if iou(h["box"], d["box"]) > 0.5), None)
        if host:
            host["frames"] += d["frames"]
        else:
            merged.append(d)
    defects = merged

    # ---- metrics ----
    matched_gt = {j for d in defects for j, g in enumerate(s.boxes) if iou(d["box"], g[:4]) > 0.3}
    real = [d for d in defects if d["gt"]]
    m = {
        "frames": n_frames, "defects": len(defects), "gt_defects": len(s.boxes),
        "gt_found": len(matched_gt), "recall": round(len(matched_gt) / max(len(s.boxes), 1), 3),
        "precision": round(len(real) / max(len(defects), 1), 3),
        "class_acc": round(float(np.mean([d["label"] == d["gt"] for d in real])), 3) if real else None,
        "real_defects": len(real),
        "ms_det_per_frame": round(1000 * t_det / max(n_frames, 1), 1),
        "frames_per_s": round(n_frames / max(t_det, 1e-9), 1),
        "rows": rows,
    }

    # ---- report per section (one row of tiles = one section of strip) ----
    sections = []
    for r_ in range(s.length // TILE):
        ds = [d for d in defects if r_ * TILE <= (d["box"][1] + d["box"][3]) / 2 < (r_ + 1) * TILE]
        sections.append({"section": r_ + 1, "defects": len(ds), "types": dict(Counter(d["label"] for d in ds)),
                         "decision": decide(ds)})

    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(s.image).convert("RGB")      # strip overview: critical types in orange
    dr = ImageDraw.Draw(im)
    for d in defects:
        dr.rectangle(d["box"], outline=(217, 89, 38) if d["label"] in RULE["critical_types"] else (201, 133, 0), width=3)
    im.save(out_dir / "strip_overview.jpg", quality=80)
    for p in out_dir.glob("crop_*.jpg"):
        p.unlink()
    ex = []
    for d in sorted(defects, key=lambda d: -d["frames"])[:12]:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        crop = Image.fromarray(s.image[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8]).convert("RGB").resize((160, 160))
        name = f"crop_{d['id']}.jpg"
        crop.save(out_dir / name, quality=85)
        ex.append({"img": name, **{k: d[k] for k in ("label", "gt", "frames", "mean_conf")}})
    res = {"metrics": m, "rule": RULE, "sections": sections, "examples": ex, "timeline": timeline,
           "defects": defects, "strip_px": [int(s.image.shape[1]), int(s.image.shape[0])], "labels_es": ES}
    (out_dir / "inspection.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(m, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models/neu_yolo11n.pt"))
    ap.add_argument("--data", default=str(ROOT / "datasets/neu"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--rows", type=int, default=12, help="strip length in tiles of 3 images")
    ap.add_argument("--conf", type=float, default=0.1)
    a = ap.parse_args()
    run(a.weights, a.data, Path(a.out), conf=a.conf, rows=a.rows)
