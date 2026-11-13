"""Cascade inspection pipeline on a moving steel strip.

  Stage 1  YOLO11n detector on EVERY frame (cheap, ~ms on CPU). Detections are associated across
           frames in strip coordinates (the line encoder gives the strip position), so each physical
           defect becomes one object, however many frames it is seen in.
  Stage 2  VLM on a FEW crops per tracked defect, only until two answers agree (consensus).
  Stage 3  Report per strip section: defects, type, severity and OK / REVISAR / RECHAZAR.

Usage:
  python src/pipeline.py --weights models/neu_yolo11n.pt --data datasets/neu --backend ollama
  python src/pipeline.py ... --backend detector     # no VLM, plumbing test only
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
import vlm

ROOT = Path(__file__).resolve().parents[1]
MAX_QUERIES, AGREE, EVERY = 4, 2, 6      # VLM budget per defect, answers that must agree, frames between queries
MARGIN = 6                               # only query when the box is fully inside the view


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def severity_proxy(box):
    frac = (box[2] - box[0]) * (box[3] - box[1]) / (TILE * TILE)
    return "high" if frac > 0.45 else "medium" if frac > 0.15 else "low"


def run(weights, data, backend, out_dir, view_h=400, step=8, conf=0.1, rows=30):
    model = YOLO(weights)
    s = build_strip(Path(data), n_rows=rows)
    tracks = defaultdict(lambda: {"boxes": [], "det": [], "conf": [], "queries": [], "last_q": -99, "done": False,
                                  "last_seen": -99, "ref": (0, 0, 0, 0)})
    n_frames, t_det, t_vlm = 0, 0.0, 0.0
    timeline = []                     # (frame, n_dets, n_vlm_calls_this_frame)
    for fi, y, fr in frames(s, view_h, step):
        n_frames += 1
        t0 = time.time()
        r = model.predict(fr, imgsz=256, conf=conf, verbose=False)[0]
        t_det += time.time() - t0
        calls = 0
        if len(r.boxes):
            for b, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                sb = (b[0], b[1] + y, b[2], b[3] + y)            # box in strip coordinates
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
                inside = b[1] > MARGIN and b[3] < view_h - MARGIN
                if (not tr["done"] and inside and fi - tr["last_q"] >= EVERY and len(tr["queries"]) < MAX_QUERIES):
                    x1, y1, x2, y2 = [int(v) for v in b]
                    pad = 8
                    crop = fr[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                    t1 = time.time()
                    ans = vlm.ask(crop, backend, Counter(tr["det"]).most_common(1)[0][0])
                    t_vlm += time.time() - t1
                    tr["queries"].append(ans); tr["last_q"] = fi; calls += 1
                    if backend != "detector":
                        print(f"[frame {fi}] VLM call {calls} · defect {tid} · {ans['label']} "
                              f"({time.time() - t1:.0f} s)", flush=True)
                    top, n = Counter(q["label"] for q in tr["queries"]).most_common(1)[0]
                    if n >= AGREE:
                        tr["done"] = True
        timeline.append((fi, 0 if r.boxes is None else len(r.boxes), calls))

    # ---- consolidate tracks ----
    defects = []
    for tid, tr in tracks.items():
        if len(tr["boxes"]) < 3:                      # flicker: ignore very short tracks
            continue
        bx = np.median(np.array(tr["boxes"]), axis=0).tolist()
        det_label = Counter(tr["det"]).most_common(1)[0][0]
        if tr["queries"]:
            final, n = Counter(q["label"] for q in tr["queries"]).most_common(1)[0]
            sev = Counter(q["severity"] for q in tr["queries"] if q["label"] == final).most_common(1)
            sev = sev[0][0] if sev else "medium"
            consensus = n >= AGREE
        else:
            final, consensus, sev = det_label, False, "medium"
        if backend != "ollama":
            sev = severity_proxy(bx)
        # ground truth match
        best = max(s.boxes, key=lambda g: iou(bx, g[:4]))
        gt = best[4] if iou(bx, best[:4]) > 0.3 else None
        defects.append({"id": tid, "box": [round(v, 1) for v in bx], "frames": len(tr["boxes"]),
                        "det_label": det_label, "final": final, "severity": sev, "consensus": consensus,
                        "vlm_calls": len(tr["queries"]), "answers": tr["queries"], "gt": gt,
                        "mean_conf": round(float(np.mean(tr["conf"])), 3)})

    # ---- merge duplicates: same physical defect split into two objects ----
    defects.sort(key=lambda d: -d["frames"])
    merged = []
    for d in defects:
        host = next((h for h in merged if iou(h["box"], d["box"]) > 0.5), None)
        if host:
            host["frames"] += d["frames"]; host["vlm_calls"] += d["vlm_calls"]
        else:
            merged.append(d)
    defects = merged

    # ---- metrics ----
    gt_boxes = s.boxes
    matched_gt = set()
    for d in defects:
        for j, g in enumerate(gt_boxes):
            if iou(d["box"], g[:4]) > 0.3:
                matched_gt.add(j)
    real = [d for d in defects if d["gt"]]
    m = {
        "frames": n_frames, "tracks": len(defects), "gt_defects": len(gt_boxes),
        "gt_found": len(matched_gt), "recall": round(len(matched_gt) / max(len(gt_boxes), 1), 3),
        "precision": round(len(real) / max(len(defects), 1), 3),
        "class_acc_detector": round(np.mean([d["det_label"] == d["gt"] for d in real]), 3) if real else None,
        "class_acc_cascade": round(np.mean([d["final"] == d["gt"] for d in real]), 3) if real else None,
        "vlm_calls": int(sum(d["vlm_calls"] for d in defects)),
        "vlm_calls_if_every_frame": int(sum(d["frames"] for d in defects)),
        "consensus_rate": round(np.mean([d["consensus"] for d in defects]), 3) if defects else None,
        "fp_suppressed_by_vlm": sum(1 for d in defects if d["final"] == "ok"),
        "ms_det_per_frame": round(1000 * t_det / max(n_frames, 1), 1),
        "s_vlm_total": round(t_vlm, 1),
        "backend": backend,
    }
    m["vlm_saving_pct"] = round(100 * (1 - m["vlm_calls"] / max(m["vlm_calls_if_every_frame"], 1)), 1)

    # ---- report per section (one row of tiles = one section of strip) ----
    rows = s.length // TILE
    sections = []
    for r_ in range(rows):
        ds = [d for d in defects if r_ * TILE <= (d["box"][1] + d["box"][3]) / 2 < (r_ + 1) * TILE and d["final"] != "ok"]
        sev = [d["severity"] for d in ds]
        # OK: no defects or only low-severity (cosmetic) ones; REVISAR: any medium; RECHAZAR: any high or >= 5 defects
        decision = ("RECHAZAR" if "high" in sev or len(ds) >= 5 else "REVISAR" if "medium" in sev else "OK")
        sections.append({"section": r_ + 1, "defects": len(ds), "types": dict(Counter(d["final"] for d in ds)),
                         "decision": decision})

    out_dir.mkdir(parents=True, exist_ok=True)
    # strip overview image with final labels
    im = Image.fromarray(s.image).convert("RGB")
    dr = ImageDraw.Draw(im)
    col = {"low": (25, 158, 112), "medium": (201, 133, 0), "high": (217, 89, 38)}
    for d in defects:
        if d["final"] != "ok":
            dr.rectangle(d["box"], outline=col.get(d["severity"], (57, 135, 229)), width=3)
    im.save(out_dir / "strip_overview.jpg", quality=80)
    # example crops
    ex = []
    for d in sorted(defects, key=lambda d: -d["frames"])[:12]:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        crop = Image.fromarray(s.image[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8]).convert("RGB").resize((160, 160))
        name = f"crop_{d['id']}.jpg"
        crop.save(out_dir / name, quality=85)
        ex.append({"img": name, **{k: d[k] for k in ("final", "det_label", "gt", "severity", "vlm_calls", "frames", "consensus")},
                   "reason": d["answers"][-1]["reason"] if d["answers"] else ""})
    res = {"metrics": m, "sections": sections, "examples": ex, "timeline": timeline,
           "defects": [{k: v for k, v in d.items() if k != "answers"} for d in defects],
           "strip_px": [int(s.image.shape[1]), int(s.image.shape[0])], "labels_es": ES}
    (out_dir / f"cascade_{backend}.json").write_text(json.dumps(res, ensure_ascii=False))
    print(json.dumps(m, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models/neu_yolo11n.pt"))
    ap.add_argument("--data", default=str(ROOT / "datasets/neu"))
    ap.add_argument("--backend", default="ollama", choices=["ollama", "detector"])
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--rows", type=int, default=30, help="strip length in tiles (fewer = fewer VLM calls)")
    ap.add_argument("--conf", type=float, default=0.1)
    a = ap.parse_args()
    run(a.weights, a.data, a.backend, Path(a.out), conf=a.conf, rows=a.rows)
