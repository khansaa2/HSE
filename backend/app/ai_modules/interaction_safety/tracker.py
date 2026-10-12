"""YOLO-Segmentation tracker for Interaction Safety."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    track_id: int
    class_name: str
    confidence: float
    bbox: list[float]
    center: list[float]
    mask_pts: list[list[float]] | None = field(default=None)


class InteractionTracker:
    def __init__(self, model_path: str, conf: float = 0.25, imgsz: int = 640, tracker: str = "bytetrack.yaml"):
        onnx_path = model_path.replace(".pt", ".onnx")
        self.model_path = onnx_path if os.path.exists(onnx_path) else model_path
        self.model = YOLO(self.model_path)
        self.names = self.model.names
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker

    def detect_and_track(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        output: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            masks_xy = result.masks.xy if result.masks is not None else None
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                class_name = str(self.names.get(cls_id, cls_id)).lower().replace(" ", "_")
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = (
                    int(box.id[0]) if box.id is not None
                    else int((x1 + x2 + y1 + y2) % 1_000_000)
                )
                mask_pts: list[list[float]] | None = None
                if masks_xy is not None and i < len(masks_xy):
                    pts = masks_xy[i]
                    if len(pts) >= 3:
                        mask_pts = [[float(p[0]), float(p[1])] for p in pts]
                output.append(Detection(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=round(conf, 3),
                    bbox=[x1, y1, x2, y2],
                    center=[(x1 + x2) / 2, (y1 + y2) / 2],
                    mask_pts=mask_pts,
                ))
        return output
