# Cascaded defect inspection: detector + local VLM

**Interactive dashboard:** [English](https://alvaromcib.github.io/inspeccion-defectos-cascada/en/) · [Español](https://alvaromcib.github.io/inspeccion-defectos-cascada/)

Visual inspection of a hot-rolled steel strip with a **cascade of models**:

1. **YOLO11n detector on every frame.** It is cheap, ~20 ms on CPU. It localises candidate defects and links them across frames using the strip position (on a real line, the line encoder).
2. **A local vision-language model (VLM) on only a few crops per defect.** It classifies type and severity and explains its decision. It is asked until **two answers agree** (at most 4 times), so there is no need to run every frame through a large model.
3. **Per-section report:** defects, types and severity, and an **OK / REVIEW / REJECT** decision.

The VLM runs **locally with Ollama**: plant images never leave the factory and there is no per-call cost.

Demo project built on the public NEU-DET dataset (surface defects of hot-rolled steel strip). No client data is used.

## Results

See `results/results.json` and the dashboard: VLM calls avoided compared with analysing every frame, defects localised, defect-type accuracy (detector vs cascade) and per-section decisions.

## How to run it

```bash
pip install -r requirements.txt
python src/get_data.py              # 1) downloads NEU-DET and converts it to YOLO format (datasets/neu)
                                    # 2) detector already trained: models/neu_yolo11n.pt
ollama pull qwen2.5vl:7b            # 3) local VLM (or qwen2.5vl:3b on a smaller machine)
python src/pipeline.py --backend ollama --rows 12   # 4) full cascade
python src/build_dashboard.py       # 5) rebuilds docs/ (ES) and docs/en/ (EN) with the VLM results
```

`--backend detector` runs the pipeline without the VLM, to test the integration. `OLLAMA_VLM` changes the model. The VLM prompt is in Spanish, so its short explanations are in Spanish.

## How it would apply on a real line

- A line-scan or area camera over the strip or part, synchronised with the encoder.
- A detector trained on the plant's own defects. The VLM adds the description and a second opinion without retraining for every new defect type.
- Integration with MES/quality systems: a report per coil or batch and traceability of each defect with its image.

---

**Álvaro Martín Coll** · Physicist · AI engineer for industry and energy · [Malt profile](https://www.malt.com/profile/alvaromartincoll)

Data: K. Song, Y. Yan, NEU surface defect database, Northeastern University (research use).
