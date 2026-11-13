"""Ask another VLM about the SAME defects found in a previous cascade run, once per defect.

It rebuilds the same strip (same seed and length), crops every defect the run kept (same 8 px padding
as pipeline.py) and asks the model once. Then it compares type accuracy on the real defects
(those matched to a label) for: the detector, the VLM of the original run, and the new VLM.
One call per defect, so it costs far less than a full pipeline run.

Usage (Windows):  set OLLAMA_VLM=qwen2.5vl:7b
                  python src/compare_vlm.py --rows 12
Output: results/vlm_compare_<model>.json and a summary on screen.
"""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

import vlm
from strip import build_strip

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "results" / "cascade_ollama.json"))
    ap.add_argument("--data", default=str(ROOT / "datasets" / "neu"))
    ap.add_argument("--rows", type=int, default=12, help="must match the run being compared")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "detector"])
    a = ap.parse_args()

    raw = Path(a.run).read_bytes()
    try:
        run = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        run = json.loads(raw.decode("cp1252"))
    model = os.getenv("OLLAMA_VLM", "qwen2.5vl:7b") if a.backend == "ollama" else "detector-echo"
    s = build_strip(Path(a.data), n_rows=a.rows)
    assert [int(s.image.shape[1]), int(s.image.shape[0])] == run["strip_px"], "strip differs from the run: check --rows"

    rows, t0 = [], time.time()
    defects = run["defects"]
    for i, d in enumerate(defects, 1):
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        g = s.image[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8]
        crop = np.stack([g, g, g], -1)
        t1 = time.time()
        ans = vlm.ask(crop, a.backend, d["det_label"])
        dt = time.time() - t1
        rows.append({"id": d["id"], "gt": d["gt"], "det_label": d["det_label"],
                     "run_vlm_label": d.get("vlm_label", d["final"]) if d["vlm_calls"] else None,
                     "new_label": ans["label"], "severity": ans["severity"], "reason": ans["reason"], "s": round(dt, 1)})
        print(f"[{i}/{len(defects)}] defect {d['id']} · {ans['label']} (ref: {d['gt'] or '-'}) ({dt:.0f} s)", flush=True)

    real = [r for r in rows if r["gt"]]
    asked = [r for r in real if r["run_vlm_label"]]
    acc = lambda rs, k: round(float(np.mean([r[k] == r["gt"] for r in rs])), 3) if rs else None
    summary = {
        "model": model, "defects": len(rows), "real_defects": len(real),
        "acc_detector": acc(real, "det_label"),
        "acc_new_vlm": acc(real, "new_label"),
        "acc_run_vlm_same_defects": acc(asked, "run_vlm_label"),
        "acc_new_vlm_same_defects": acc(asked, "new_label"),
        "acc_detector_same_defects": acc(asked, "det_label"),
        "new_vlm_answers": dict(Counter(r["new_label"] for r in rows)),
        "s_per_call": round(float(np.mean([r["s"] for r in rows])), 1),
        "minutes_total": round((time.time() - t0) / 60, 1),
    }
    out = ROOT / "results" / f"vlm_compare_{model.replace(':', '_')}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
