"""Re-apply the cascade's fusion rule to a finished VLM run, without calling the VLM again.

The run in results/cascade_ollama.json (qwen2.5vl:3b, CPU) used the first design, where the VLM's
consensus label decided the defect type. That design classified 51.1% of real defects correctly,
against 95.7% for the detector alone: the 3B model answered "patches" for almost everything.

The current design (also in pipeline.py) lets the detector decide the type and keeps the VLM for what
it adds: severity, a natural-language explanation, and a veto when it agrees the crop has no defect.
This script recomputes the final labels, metrics and per-section report with that rule, keeping the
VLM's answers (label, severity, explanation) exactly as recorded in the run.

Usage: python src/fuse.py   ->  results/cascade_ollama_fused.json
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TILE = 200  # same as strip.TILE (kept here so this script needs no dataset or model)


def load(path: Path) -> dict:
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:          # runs written on Windows before the utf-8 fix
        return json.loads(raw.decode("cp1252"))


def main():
    src = ROOT / "results" / "cascade_ollama.json"
    res = load(src)
    defects = res["defects"]
    for d in defects:
        d["vlm_label"] = d.get("vlm_label", d["final"]) if d["vlm_calls"] else None
        d["final"] = "ok" if (d["vlm_label"] == "ok" and d["consensus"]) else d["det_label"]

    real = [d for d in defects if d["gt"]]
    m = res["metrics"]
    m["class_acc_vlm_type"] = m["class_acc_cascade"]          # first design: VLM decided the type
    m["class_acc_cascade"] = round(float(np.mean([d["final"] == d["gt"] for d in real])), 3)
    m["fp_suppressed_by_vlm"] = sum(1 for d in defects if d["final"] == "ok")
    # count every VLM call actually made (the run's metric left out calls on short tracks discarded as flicker)
    m["vlm_calls"] = int(sum(t[2] for t in res["timeline"]))
    m["vlm_saving_pct"] = round(100 * (1 - m["vlm_calls"] / max(m["vlm_calls_if_every_frame"], 1)), 1)
    m["fusion"] = "detector_type+vlm_severity_veto"
    m["vlm_model"] = "qwen2.5vl:3b"
    m["vlm_type_answers"] = dict(Counter(d["vlm_label"] for d in defects if d["vlm_label"]))

    rows = res["strip_px"][1] // TILE
    sections = []
    for r in range(rows):
        ds = [d for d in defects if r * TILE <= (d["box"][1] + d["box"][3]) / 2 < (r + 1) * TILE and d["final"] != "ok"]
        sev = [d["severity"] for d in ds]
        decision = "RECHAZAR" if "high" in sev or len(ds) >= 5 else "REVISAR" if "medium" in sev else "OK"
        sections.append({"section": r + 1, "defects": len(ds), "types": dict(Counter(d["final"] for d in ds)),
                         "decision": decision})
    res["sections"] = sections

    by_id = {d["id"]: d for d in defects}
    for e in res["examples"]:
        d = by_id[int(e["img"].split("_")[1].split(".")[0])]
        e["vlm_label"], e["final"] = d["vlm_label"], d["final"]

    out = ROOT / "results" / "cascade_ollama_fused.json"
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: m[k] for k in ("class_acc_detector", "class_acc_vlm_type", "class_acc_cascade",
                                        "fp_suppressed_by_vlm", "vlm_calls", "vlm_saving_pct")}, indent=1))
    print("sections:", dict(Counter(s["decision"] for s in sections)), "->", out.name)


if __name__ == "__main__":
    main()
