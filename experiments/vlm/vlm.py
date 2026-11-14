"""Stage 2: vision-language model that confirms and describes each defect from a crop.

Backends
  - "ollama"   : local VLM (default qwen2.5vl:7b). Images never leave the machine.
  - "detector" : offline fallback, echoes the detector class. Lets the pipeline run end-to-end
                 without a VLM (used to test plumbing; NOT a second opinion).
"""
import base64
import io
import json
import os
import urllib.request

from PIL import Image

from strip import CLASSES

PROMPT = """Eres inspector de calidad en una línea de laminación en caliente.
La imagen es un recorte en escala de grises de la superficie de una banda de acero.
Clasifica el defecto principal en UNA de estas categorías:
crazing (grietas en red), inclusion (inclusión de material extraño), patches (manchas/zonas irregulares),
pitted_surface (picaduras), rolled-in_scale (cascarilla laminada), scratches (arañazos lineales), ok (sin defecto real).
Valora la severidad: low, medium o high.
Responde SOLO con JSON: {"label": "<categoría>", "severity": "<low|medium|high>", "reason": "<máx. 15 palabras>"}"""


def _b64(crop) -> str:
    buf = io.BytesIO()
    Image.fromarray(crop).resize((224, 224)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask_ollama(crop, model=None, host=None) -> dict:
    model = model or os.getenv("OLLAMA_VLM", "qwen2.5vl:7b")
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    body = {"model": model, "stream": False, "format": "json", "options": {"temperature": 0},
            "messages": [{"role": "user", "content": PROMPT, "images": [_b64(crop)]}]}
    req = urllib.request.Request(f"{host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=int(os.getenv("OLLAMA_TIMEOUT", "600"))).read())["message"]["content"]
    ans = json.loads(out)
    label = ans.get("label", "").strip().lower()
    return {"label": label if label in CLASSES + ["ok"] else "unknown",
            "severity": ans.get("severity", "medium"), "reason": ans.get("reason", "")}


def ask(crop, backend: str, det_label: str) -> dict:
    if backend == "ollama":
        return ask_ollama(crop)
    return {"label": det_label, "severity": "medium", "reason": "modo sin VLM (eco del detector)"}
