"""PPE Detection module — migrated from backend-epi-main."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


class PPEDetector:
    def __init__(self, model_path: str | Path) -> None:
        self.model = YOLO(str(model_path))
        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

    @property
    def names(self) -> dict[int, str]:
        return self.model.names

    def parse_detections(self, results, names: dict) -> list[dict[str, Any]]:
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(r.boxes.conf[i])
                cls = int(r.boxes.cls[i])
                tid = int(r.boxes.id[i]) if r.boxes.id is not None else -1
                cls_name = names.get(cls, f"cls{cls}")
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 2),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "track_id": tid,
                    "violation": cls_name.lower().startswith("no "),
                })
        return detections

    def track(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model.track(frame, conf=0.3, persist=True, imgsz=640, verbose=False)
        return self.parse_detections(results, self.model.names)


# Singleton per process
_detector: PPEDetector | None = None


def get_detector(model_path: str | Path) -> PPEDetector:
    global _detector
    if _detector is None:
        _detector = PPEDetector(model_path)
    return _detector
