"""Intrusion / Voie Engins detection — migrated from backend-zones-dangereuses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from app.utils.geometry import point_in_polygon

DEFAULT_POLYGON = [
    [200, 400], [800, 400], [900, 700], [100, 700],
]

# Colours (BGR)
_COLORS = {
    "person_safe":    (34,  197,  94),
    "person_lane":    (  0,   0, 230),
    "vehicle":        (249, 115,  22),
    "operator":       ( 56, 189, 248),
}


class IntrusionDetector:
    def __init__(self, model_path: str | Path) -> None:
        self.model = YOLO(str(model_path))
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model(frame, conf=0.25, imgsz=640, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            cls_name = results.names.get(cls, str(cls))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
            })
        return detections


class MachineLaneMonitor:
    def __init__(self, polygon_points: list[list[float]] | None = None) -> None:
        self.polygon = polygon_points or DEFAULT_POLYGON
        self.intrusions_total: int = 0
        self._prev_in_lane: set = set()

    def analyze(self, frame_shape: tuple, detections: list[dict]) -> dict[str, Any]:
        h, w = frame_shape[:2]
        scale_x = w / 1280.0
        scale_y = h / 720.0

        scaled_polygon = [
            [pt[0] * scale_x, pt[1] * scale_y]
            for pt in self.polygon
        ]

        person_classes = {"person", "operator"}
        persons_in_lane: list[dict] = []
        alerts: list[dict] = []

        for det in detections:
            if det["class"] not in person_classes:
                continue
            cx, cy = det["center"]
            if point_in_polygon((cx, cy), scaled_polygon):
                persons_in_lane.append(det)

        person_in_lane = len(persons_in_lane) > 0

        # Count new intrusions
        current_ids = {id(d) for d in persons_in_lane}
        new_intrusions = current_ids - self._prev_in_lane
        self.intrusions_total += len(new_intrusions)
        self._prev_in_lane = current_ids

        if person_in_lane:
            alerts.append({
                "level": "CRITIQUE",
                "message": f"ALERTE: {len(persons_in_lane)} personne(s) dans la voie engins !",
            })
        elif detections:
            vehicle_near = any(
                d["class"] in {"forklift", "crane", "excavator", "dump_truck"}
                for d in detections
            )
            if vehicle_near:
                alerts.append({
                    "level": "DANGER",
                    "message": "Engin détecté dans la zone surveillée",
                })

        return {
            "person_in_lane": person_in_lane,
            "persons_in_lane": persons_in_lane,
            "scaled_polygon": scaled_polygon,
            "alerts": alerts,
            "intrusions_total": self.intrusions_total,
        }


def annotate_frame(
    frame: np.ndarray,
    detections: list[dict],
    result: dict,
    frame_number: int,
    total_frames: int,
    fps: float,
    show_polygon: bool = True,
) -> dict:
    import base64

    alerts = result["alerts"]
    person_in_lane = result["person_in_lane"]
    persons_in_lane = result["persons_in_lane"]
    scaled_polygon = result["scaled_polygon"]
    intrusions_total = result.get("intrusions_total", 0)

    w_frame = frame.shape[1]
    scale_base = w_frame / 1280.0

    # Draw polygon
    if show_polygon and scaled_polygon:
        pts = np.array(scaled_polygon, dtype=np.int32)
        overlay = frame.copy()
        fill = (0, 0, 150) if person_in_lane else _COLORS["person_safe"]
        border = fill
        cv2.fillPoly(overlay, [pts], fill)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=border, thickness=1, lineType=cv2.LINE_AA)
        font_zone = max(0.3, 0.38 * scale_base)
        cv2.putText(
            frame, "VOIE ENGINS",
            (int(scaled_polygon[0][0]) + 4, int(scaled_polygon[0][1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_zone, border, 1, cv2.LINE_AA,
        )

    # Draw bboxes
    font_label = max(0.28, 0.32 * scale_base)
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        cls = det["class"]
        if cls in ("person", "operator"):
            in_lane = bool(scaled_polygon) and point_in_polygon(det["center"], scaled_polygon)
            color = _COLORS["person_lane"] if in_lane else _COLORS["person_safe"]
        elif cls in ("forklift", "crane", "excavator", "dump_truck"):
            color = _COLORS["vehicle"]
        else:
            color = (160, 160, 160)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
        label = cls
        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_label, 1)
        label_y = max(lh + bl + 2, y1)
        cv2.rectangle(frame, (x1, label_y - lh - bl - 2), (x1 + lw + 3, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 1, label_y - bl - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_label, (255, 255, 255), 1, cv2.LINE_AA)

    # Alert banners
    font_alert = max(0.35, 0.40 * scale_base)
    y = 18
    for alert in alerts:
        level = alert["level"]
        bg = (0, 0, 220) if level == "CRITIQUE" else (0, 60, 200)
        text = alert["message"]
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_alert, 1)
        banner = frame.copy()
        cv2.rectangle(banner, (6, y - th - bl - 3), (10 + tw + 6, y + 3), bg, -1)
        cv2.addWeighted(banner, 0.82, frame, 0.18, 0, frame)
        cv2.putText(frame, text, (9, y), cv2.FONT_HERSHEY_SIMPLEX, font_alert, (255, 255, 255), 1, cv2.LINE_AA)
        y += th + bl + 7

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return {
        "type": "frame",
        "frame": base64.b64encode(buf).decode("utf-8"),
        "alerts": alerts,
        "intrusions_total": intrusions_total,
        "frame_number": frame_number,
        "total_frames": total_frames,
        "fps": fps,
        "persons_in_zone": len(persons_in_lane),
    }


_detector: IntrusionDetector | None = None


def get_detector(model_path: str | Path) -> IntrusionDetector:
    global _detector
    if _detector is None:
        _detector = IntrusionDetector(model_path)
    return _detector
