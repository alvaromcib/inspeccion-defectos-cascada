# Cascaded defect inspection: detector + local VLM

**Interactive dashboard:** [English](https://alvaromcib.github.io/inspeccion-defectos-cascada/en/) · [Español](https://alvaromcib.github.io/inspeccion-defectos-cascada/)

Visual inspection of a hot-rolled steel strip with a **cascade of models**:

1. **YOLO11n detector on every frame.** It is cheap (~80 ms per frame on a laptop CPU). It localises candidate defects, decides their **type**, and links them across frames using the strip position (on a real line, the line encoder).
2. **A local vision-language model (VLM) on only a few crops per defect.** It grades **severity**, writes a short **explanation** for the quality report and can **veto** a detection it agrees is not a defect. It is asked until **two answers agree** (at most 4 times), so there is no need to run every frame through a large model.
3. **Per-section report:** defects, types and severity, and an **OK / REVIEW / REJECT** decision.

The VLM runs **locally with Ollama**: plant images never leave the factory and there is no per-call cost.

Demo project built on the public NEU-DET dataset (surface defects of hot-rolled steel strip). No client data is used.

## Results

Real run on a laptop without a GPU: qwen2.5vl:3b with Ollama, 251 frames, 12 strip sections (`results/cascade_ollama.json`; every VLM call with its duration in `results/vlm_run.log`).

| Metric | Value |
|---|---|
| VLM calls | 114 instead of 1,927 if it looked at every frame (**−94%**) |
| Defect type correct (detector decides) | **95.7%** |
| Defect type correct if the 3B VLM decided it | 51.1% |
| Defects with consensus between VLM answers | 80% |
| Labelled defects localised by the detector | 45 of 86 (52%) |
| VLM time | ≈80 s per call on CPU, ~2.4 h in total |

### What didn't work, and the design change

The first design let the VLM's consensus label decide the defect type. With a 3B model on CPU it answered "patches" for 45 of the 50 defects it was asked about, so the cascade classified **worse** than the detector (51.1% vs 95.7%). Consensus can't fix that: two identical wrong answers are still consensus.

The final design keeps the VLM for what it adds (severity, explanation, veto) and lets the detector, which was trained for it, decide the type. `src/fuse.py` re-applies that rule to the same run without calling the VLM again (`results/cascade_ollama_fused.json`, which feeds the dashboard); `src/pipeline.py` uses it for new runs. A larger VLM (e.g. qwen2.5vl:7b on a GPU) or a better prompt can be measured against the detector with the same metrics before giving it the type decision.

## How to run it

```bash
pip install -r requirements.txt
python src/get_data.py              # 1) downloads NEU-DET and converts it to YOLO format (datasets/neu)
                                    # 2) detector already trained: models/neu_yolo11n.pt
ollama pull qwen2.5vl:3b            # 3) local VLM (the run above used the 3B model)
python src/pipeline.py --backend ollama --rows 12   # 4) full cascade (OLLAMA_VLM=qwen2.5vl:3b to use the 3B model)
python src/fuse.py                  # 5) only for runs made with the first design: re-applies the fusion rule
python src/build_dashboard.py       # 6) rebuilds docs/ (ES) and docs/en/ (EN) with the VLM results
```

`--backend detector` runs the pipeline without the VLM, to test the integration. `OLLAMA_VLM` changes the model. The VLM prompt is in Spanish, so its explanations are in Spanish. Each VLM call is printed with its duration; `OLLAMA_TIMEOUT` (default 600 s) bounds a single call.

## How it would apply on a real line

- A line-scan or area camera over the strip or part, synchronised with the encoder.
- A detector trained on the plant's own defects. The VLM adds the description and a second opinion without retraining for every new defect type.
- Integration with MES/quality systems: a report per coil or batch and traceability of each defect with its image.

---

**Álvaro Martín Coll** · Physicist · AI engineer for industry and energy · [Malt profile](https://www.malt.com/profile/alvaromartincoll)

Data: K. Song, Y. Yan, NEU surface defect database, Northeastern University (research use).
