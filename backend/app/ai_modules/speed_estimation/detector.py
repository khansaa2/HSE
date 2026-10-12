"""Vehicle Speed Estimation — migrated from backend-speed-main."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


class SpeedEstimator:
    def __init__(self, model: YOLO) -> None:
        self.model = model
        self.track_state: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "last_world": None,
                "last_time_s": None,
                "speeds_kmh": deque(maxlen=6),
                "speed_kmh": 0.0,
                "trajectory": deque(maxlen=25),
            }
        )

    @staticmethod
    def allowed_classes(names: dict[int, str]) -> set[int]:
        labels = {
            "car", "truck", "bus", "forklift", "vehicle", "van", "pickup",
            "excavator", "crane", "dump_truck", "wheel_loader", "loader",
        }
        ids = {i for i, n in names.items() if str(n).lower() in labels}
        if ids:
            return ids
        fuzzy = ("car", "truck", "bus", "forklift", "vehicle", "van", "pickup", "excavator", "crane", "loader", "dump")
        ids = {i for i, n in names.items() if any(k in str(n).lower() for k in fuzzy)}
        if ids:
            return ids
        skip = {"person", "operator", "worker", "pedestrian"}
        ids = {i for i, n in names.items() if str(n).lower() not in skip}
        return ids if ids else set(names.keys())

    @staticmethod
    def compute_homography(image_points: list[list[float]], world_points: list[list[float]]) -> np.ndarray:
        ip = np.array(image_points, dtype=np.float32)
        wp = np.array(world_points, dtype=np.float32)
        if ip.shape != (4, 2) or wp.shape != (4, 2):
            raise ValueError("image_points/world_points must be 4x2.")
        h, _ = cv2.findHomography(ip, wp)
        if h is None:
            raise ValueError("Could not compute homography.")
        return h

    @staticmethod
    def ground_point(box: np.ndarray) -> tuple[float, float]:
        x1, y1, x2, y2 = box.tolist()
        return (x1 + x2) / 2.0, y2

    @staticmethod
    def px_to_world(h: np.ndarray, cx: float, cy: float) -> np.ndarray:
        p = np.array([[[cx, cy]]], dtype=np.float32)
        return cv2.perspectiveTransform(p, h)[0][0]

    def update_speed(self, tid: int, world: np.ndarray, now_s: float) -> float:
        state = self.track_state[tid]
        prev_world = state["last_world"]
        prev_time = state["last_time_s"]
        if prev_world is None or prev_time is None:
            state["last_world"] = world
            state["last_time_s"] = now_s
            return 0.0
        dt = now_s - float(prev_time)
        if dt <= 0:
            return float(state["speed_kmh"])
        dist_m = float(np.linalg.norm(world - prev_world))
        state["last_world"] = world
        state["last_time_s"] = now_s
        if dist_m < 0.02:
            return float(state["speed_kmh"])
        speed_kmh = (dist_m / dt) * 3.6
        if speed_kmh > 150.0:
            return float(state["speed_kmh"])
        state["speeds_kmh"].append(speed_kmh)
        smooth = float(np.mean(state["speeds_kmh"]))
        state["speed_kmh"] = smooth
        return smooth


def get_detector(model_path: str | Path) -> YOLO:
    model = YOLO(str(model_path))
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(source=dummy, verbose=False, imgsz=640)
    return model


_model: YOLO | None = None


def get_model(model_path: str | Path) -> YOLO:
    global _model
    if _model is None:
        _model = get_detector(model_path)
    return _model
