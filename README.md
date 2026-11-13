# Inspección de defectos en cascada: detector + VLM local

**Dashboard interactivo:** https://alvaromcib.github.io/inspeccion-defectos-cascada/

Inspección visual de una banda de acero laminado en caliente con una **cascada de modelos**:

1. **Detector YOLO11n en cada fotograma.** Es barato, ~20 ms en CPU. Localiza posibles defectos y los asocia entre fotogramas con la posición de la banda (en planta, el encoder de la línea).
2. **Modelo de visión-lenguaje (VLM) local solo en unos pocos recortes por defecto.** Clasifica tipo y severidad y explica su decisión. Se le pregunta hasta que **dos respuestas coinciden** (como máximo 4 veces), así que no hay que pasar cada fotograma por un modelo grande.
3. **Informe por tramo:** defectos, tipos y severidad, y la decisión **OK / REVISAR / RECHAZAR**.

El VLM se ejecuta con **Ollama en local**: las imágenes de planta no salen de la fábrica y no hay coste por llamada.

Proyecto demostrativo con el dataset público NEU-DET (defectos superficiales de banda laminada en caliente). No usa datos de clientes.

## Resultados

Ver `results/results.json` y el dashboard: llamadas al VLM evitadas frente a analizar cada fotograma, defectos localizados, precisión del tipo de defecto (detector frente a cascada) y decisiones por tramo.

## Ejecutarlo

```bash
pip install -r requirements.txt
python src/get_data.py              # 1) descarga NEU-DET y lo convierte a YOLO (datasets/neu)
                                    # 2) detector ya entrenado: models/neu_yolo11n.pt
ollama pull qwen2.5vl:7b            # 3) VLM local (o qwen2.5vl:3b si el equipo va justo)
python src/pipeline.py --backend ollama --rows 12   # 4) cascada completa
python src/build_dashboard.py       # 5) regenera el dashboard con los resultados del VLM
```

`--backend detector` ejecuta el pipeline sin VLM, para probar la integración. Con `OLLAMA_VLM` se cambia el modelo.

## Cómo se aplicaría en una línea real

- Cámara de línea o matricial sobre la banda o la pieza, sincronizada con el encoder.
- Detector entrenado con defectos propios. El VLM aporta la descripción y la segunda opinión sin reentrenar para cada tipo nuevo.
- Integración con MES/calidad: informe por bobina o lote y trazabilidad de cada defecto con su imagen.

---

**Álvaro Martín Coll** · Físico · Ingeniero de IA para industria y energía · [Perfil en Malt](https://www.malt.com/profile/alvaromartincoll)

Datos: K. Song, Y. Yan, NEU surface defect database, Northeastern University (uso de investigación).
