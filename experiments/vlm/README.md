# Discarded experiment: a vision-language model (VLM) as a second stage

**Idea.** A cascade where the YOLO detector runs on every frame and a local VLM (Qwen2.5-VL with Ollama) looks at only a few crops of each defect, until two answers agree, to classify it, grade its severity and explain it in natural language.

**Result.** On the same 43 defects of one strip run (12 sections, NEU-DET validation images), type accuracy was:

| Who decides the type | Correct |
|---|---|
| YOLO11n detector | 95.3% |
| qwen2.5vl:3b, cascade with consensus (2–4 calls per defect) | 46.5% |
| qwen2.5vl:7b, one call per defect | 41.9% |

Both models answered "patches" for most defects (3B: 45 of 50; 7B: 35 of 55, plus 13 answers outside the label set). Consensus can't fix a consistent bias, a larger model didn't fix it either, and the explanations sometimes contradicted the detector's type. On a laptop CPU each call took ≈77 s (3B) and ≈118 s (7B). It added cost and complexity and made the system worse, so it was **removed from the pipeline**.

## Files

- `results/run_3b_cascade.json`: full cascade run with qwen2.5vl:3b (defects, timeline, metrics). `class_acc_cascade` there is the accuracy when the VLM decided the type.
- `results/run_3b.log`: every VLM call of that run with its duration.
- `results/compare_7b.json` / `results/compare_7b.log`: qwen2.5vl:7b asked once about each defect of that run.
- `summary.json`: the numbers above, used by the dashboard.
- `vlm.py`: the Ollama client and prompt (in Spanish). `compare_vlm.py`: asks any VLM about the defects of a run.

## Reproduce

```bash
ollama pull qwen2.5vl:7b
OLLAMA_VLM=qwen2.5vl:7b python experiments/vlm/compare_vlm.py --rows 12    # ~2 h on a laptop CPU
```

The full cascade (3B run) used the pipeline as of commit `86e7d89` (`git checkout 86e7d89 -- src/pipeline.py src/vlm.py` and `python src/pipeline.py --backend ollama --rows 12`).
