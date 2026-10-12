"""Constants for Interaction Safety module."""

from __future__ import annotations
from dataclasses import dataclass

PERSON_CLASSES  = {"person", "worker", "operator"}
AT_RISK_CLASSES = {"person", "worker"}
MACHINE_CLASSES = {"forklift", "excavator", "wheel_loader", "truck", "loader", "dump_truck", "crane"}


@dataclass(frozen=True)
class ModuleConfig:
    model_path: str = "models/bestse.pt"
    max_width: int = 1280
    yolo_conf: float = 0.25
    yolo_imgsz: int = 640
    pixel_to_meter: float = 0.02
    speed_threshold_m_s: float = 0.0
    danger_distance_default_m: float = 2.5
    excavator_radius_default_m: float = 3.0
    default_skip: int = 2


ENGINE_DISTANCE_THRESHOLDS_M = {
    "forklift": 2.0,
    "wheel_loader": 3.0,
    "loader": 3.0,
    "dump_truck": 3.0,
    "truck": 3.0,
    "crane": 4.0,
}
