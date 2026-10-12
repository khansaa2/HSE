"""Smart Parking module — migrated from backend-parking-main."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box


@dataclass
class ParkingSpot:
    spot_id: str
    polygon: np.ndarray


class ParkingDetector:
    def __init__(self, model_path: Path, calibration_path: Path) -> None:
        onnx_path = model_path.with_suffix(".onnx")
        self.model = YOLO(str(onnx_path) if onnx_path.exists() else str(model_path))
        self.spots, self.calibration_size = self._parse_calibration(calibration_path)
        self._polygon_cache: dict[tuple[int, int, str], np.ndarray] = {}

    def load_calibration(self, calibration_path: Path) -> None:
        self.spots, self.calibration_size = self._parse_calibration(calibration_path)
        self._polygon_cache.clear()

    @staticmethod
    def _parse_calibration(path: Path) -> tuple[list[ParkingSpot], tuple[int, int]]:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        original_w, original_h = raw.get("original_image_size", [1920, 1080])
        spots: list[ParkingSpot] = []
        for place in raw.get("places", []):
            spots.append(ParkingSpot(
                spot_id=place["id"],
                polygon=np.array(place["polygon"], dtype=np.float32),
            ))
        return spots, (int(original_w), int(original_h))

    @staticmethod
    def _vehicle_like_labels(names: dict[int, str]) -> set[int]:
        allowed = {"car", "truck", "bus", "motorbike", "motorcycle", "vehicle"}
        matched = {idx for idx, label in names.items() if str(label).lower() in allowed}
        return matched if matched else set(names.keys())

    def _scaled_polygon(self, polygon: np.ndarray, frame_w: int, frame_h: int, spot_id: str) -> np.ndarray:
        cache_key = (frame_w, frame_h, spot_id)
        if cache_key in self._polygon_cache:
            return self._polygon_cache[cache_key]
        base_w, base_h = self.calibration_size
        scaled = polygon.copy()
        scaled[:, 0] *= frame_w / float(base_w)
        scaled[:, 1] *= frame_h / float(base_h)
        self._polygon_cache[cache_key] = scaled
        return scaled

    @staticmethod
    def _is_box_in_spot(box: tuple[float, float, float, float], polygon: np.ndarray, overlap_threshold: float = 0.3) -> bool:
        x1, y1, x2, y2 = box
        vehicle_poly = shapely_box(x1, y1, x2, y2)
        spot_poly = ShapelyPolygon(polygon.tolist())
        if not spot_poly.is_valid:
            spot_poly = spot_poly.buffer(0)
        if spot_poly.is_empty or vehicle_poly.is_empty:
            return False
        try:
            if spot_poly.intersects(vehicle_poly):
                inter_area = spot_poly.intersection(vehicle_poly).area
                if spot_poly.area > 0 and inter_area / spot_poly.area >= overlap_threshold:
                    return True
        except Exception:
            pass
        return False

    def detect(self, frame: np.ndarray, spot_ids: list[str] | None = None) -> list[dict[str, Any]]:
        frame_h, frame_w = frame.shape[:2]
        selected_ids = set(spot_ids or [s.spot_id for s in self.spots])
        selected_spots = [s for s in self.spots if s.spot_id in selected_ids]

        result = self.model.predict(source=frame, verbose=False, conf=0.35, imgsz=640)[0]
        names = {int(k): str(v) for k, v in result.names.items()}
        allowed_classes = self._vehicle_like_labels(names)

        vehicle_boxes: list[tuple[tuple[float, float, float, float], float]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for b in result.boxes:
                cls_idx = int(b.cls.item())
                if cls_idx not in allowed_classes:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                vehicle_boxes.append(((x1, y1, x2, y2), float(b.conf.item())))

        detections: list[dict[str, Any]] = []
        for spot in selected_spots:
            poly = self._scaled_polygon(spot.polygon, frame_w, frame_h, spot.spot_id)
            occupied = False
            best_conf = 0.0
            for box, conf in vehicle_boxes:
                if self._is_box_in_spot(box, poly):
                    occupied = True
                    best_conf = max(best_conf, conf)
            detections.append({
                "spotId": spot.spot_id,
                "occupied": occupied,
                "confidence": round(best_conf if occupied else 0.95, 3),
                "alert": "Spot occupé" if occupied else None,
            })
        return detections


_detector: ParkingDetector | None = None


def get_detector(model_path: Path, calibration_path: Path) -> ParkingDetector:
    global _detector
    if _detector is None:
        _detector = ParkingDetector(model_path, calibration_path)
    return _detector
