"""Simulated inspection camera over a moving hot-rolled steel strip.

Validation images (never seen in training) are tiled into a long strip, 3 tiles wide.
The camera looks at a fixed window while the strip advances, so every defect crosses
the field of view over many consecutive frames, as on a real line.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
ES = {"crazing": "Grietas", "inclusion": "Inclusión", "patches": "Manchas", "pitted_surface": "Picaduras",
      "rolled-in_scale": "Cascarilla laminada", "scratches": "Arañazos", "ok": "Sin defecto"}
TILE, COLS = 200, 3


@dataclass
class Strip:
    image: np.ndarray            # H x W grayscale
    boxes: list                  # [(x1, y1, x2, y2, cls, tile_name)] in strip coordinates

    @property
    def length(self):
        return self.image.shape[0]


def build_strip(data_dir: Path, n_rows: int = 30, seed: int = 1) -> Strip:
    imgs = sorted((data_dir / "images" / "val").glob("*.jpg"))
    rng = np.random.default_rng(seed)
    pick = rng.choice(imgs, n_rows * COLS, replace=False)
    canvas = np.zeros((n_rows * TILE, COLS * TILE), np.uint8)
    boxes = []
    for i, p in enumerate(pick):
        r, c = divmod(i, COLS)
        im = np.array(Image.open(p).convert("L").resize((TILE, TILE)))
        canvas[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = im
        lbl = data_dir / "labels" / "val" / (p.stem + ".txt")
        for line in lbl.read_text().splitlines():
            k, cx, cy, w, h = line.split()
            cx, cy, w, h = (float(v) * TILE for v in (cx, cy, w, h))
            boxes.append((c * TILE + cx - w / 2, r * TILE + cy - h / 2, c * TILE + cx + w / 2,
                          r * TILE + cy + h / 2, CLASSES[int(k)], p.stem))
    return Strip(canvas, boxes)


def frames(strip: Strip, view_h: int = 400, step: int = 8):
    """Yield (frame_index, y_offset, RGB frame). The strip moves `step` px per frame."""
    for i, y in enumerate(range(0, strip.length - view_h + 1, step)):
        g = strip.image[y:y + view_h]
        yield i, y, np.stack([g, g, g], -1)
