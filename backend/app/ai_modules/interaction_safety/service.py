"""Interaction Safety pipeline — migrated from backend-interaction-safety."""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .constants import MACHINE_CLASSES, ModuleConfig, PERSON_CLASSES
from .rules import InteractionSafetyMonitor, RuntimeRuleConfig
from .tracker import Detection, InteractionTracker

_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person":       ( 40, 200,  80),
    "worker":       ( 20, 180,  60),
    "operator":     ( 60, 220, 100),
    "forklift":     (  0, 140, 255),
    "excavator":    (  0,  60, 200),
    "wheel_loader": ( 30, 200, 220),
    "truck":        (200, 160,  30),
    "loader":       (200,  60, 200),
    "dump_truck":   (180,  40, 120),
    "crane":        (120, 210,   0),
}
_COLOR_DEFAULT   = (180, 180, 180)
_COLOR_DANGER    = (  0,   0, 255)
_COLOR_PROXIMITY = (  0, 140, 255)
_COLOR_FEET      = (  0, 255, 120)
_COLOR_NEAREST   = (  0, 100, 255)
_COLOR_DIST      = (  0, 255, 255)


def _class_color(class_name: str) -> tuple[int, int, int]:
    return _CLASS_COLORS.get(class_name, _COLOR_DEFAULT)


def _bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


class InteractionSafetyService:
    def __init__(self, config: ModuleConfig | None = None):
        self.cfg = config or ModuleConfig()
        self.tracker = InteractionTracker(
            model_path=self.cfg.model_path,
            conf=self.cfg.yolo_conf,
            imgsz=self.cfg.yolo_imgsz,
            tracker="bytetrack.yaml",
        )

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.cfg.max_width:
            return frame
        scale = self.cfg.max_width / float(w)
        return cv2.resize(frame, (self.cfg.max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    def _draw(self, frame: np.ndarray, detections: list[Detection], analysis: dict, runtime_cfg: RuntimeRuleConfig) -> np.ndarray:
        h, w = frame.shape[:2]
        eff_p2m = max(analysis.get("pixel_to_meter", runtime_cfg.pixel_to_meter), 1e-5)
        person_alert: dict[int, str] = {}
        for pair in analysis.get("danger_pairs", []):
            pid = pair["person_id"]
            level = pair.get("alert_level", "PROXIMITE")
            if pid not in person_alert or level == "CONTACT_CRITIQUE":
                person_alert[pid] = level

        def _person_color(d: Detection) -> tuple[int, int, int]:
            level = person_alert.get(d.track_id)
            if level == "CONTACT_CRITIQUE":
                return _COLOR_DANGER
            if level == "PROXIMITE":
                return _COLOR_PROXIMITY
            return _class_color(d.class_name)

        overlay = frame.copy()
        for d in detections:
            if not d.mask_pts or len(d.mask_pts) < 3:
                continue
            is_person = d.class_name in PERSON_CLASSES
            is_machine = d.class_name in MACHINE_CLASSES
            if not is_person and not is_machine:
                continue
            color = _person_color(d) if is_person else _class_color(d.class_name)
            pts = np.array(d.mask_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

        for d in detections:
            is_person = d.class_name in PERSON_CLASSES
            is_machine = d.class_name in MACHINE_CLASSES
            if not is_person and not is_machine:
                continue
            color = _person_color(d) if is_person else _class_color(d.class_name)
            in_danger = is_person and d.track_id in person_alert
            box_thick = 3 if in_danger else 2
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            level_tag = ""
            if in_danger:
                level_tag = "[CC] " if person_alert[d.track_id] == "CONTACT_CRITIQUE" else "[PR] "
            label = f"{level_tag}{d.class_name} #{d.track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, 0.45, 1)
            lx, ly = x1, max(th + baseline + 4, y1 - 2)
            cv2.rectangle(frame, (lx, ly - th - baseline - 2), (lx + tw + 6, ly + 2), color, -1)
            cv2.putText(frame, label, (lx + 3, ly - baseline), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if d.mask_pts and len(d.mask_pts) >= 3:
                pts = np.array(d.mask_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            if is_person:
                if d.mask_pts and len(d.mask_pts) >= 3:
                    bottom = max(d.mask_pts, key=lambda p: p[1])
                    fx, fy = int(bottom[0]), int(bottom[1])
                else:
                    fx = int((d.bbox[0] + d.bbox[2]) / 2)
                    fy = int(d.bbox[3])
                pt_color = color if in_danger else _COLOR_FEET
                cv2.circle(frame, (fx, fy), 6, pt_color, -1)
                cv2.circle(frame, (fx, fy), 7, (255, 255, 255), 1)

        for pair in analysis.get("danger_pairs", []):
            dist_m = pair["distance_m"]
            p_ref = pair.get("person_feet")
            m_ref = pair.get("machine_nearest_pt")
            if not p_ref or not m_ref:
                continue
            level = pair.get("alert_level", "PROXIMITE")
            line_color = _COLOR_DANGER if level == "CONTACT_CRITIQUE" else _COLOR_PROXIMITY
            pf = (int(p_ref[0]), int(p_ref[1]))
            mn = (int(m_ref[0]), int(m_ref[1]))
            cv2.line(frame, pf, mn, line_color, 2, cv2.LINE_AA)
            cv2.circle(frame, mn, 5, _COLOR_NEAREST, -1)
            mid_x = int((pf[0] + mn[0]) / 2)
            mid_y = int((pf[1] + mn[1]) / 2)
            dist_label = "CONTACT" if dist_m == 0.0 else f"{dist_m:.2f} m"
            (dw, dh), _ = cv2.getTextSize(dist_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (mid_x + 3, mid_y - dh - 6), (mid_x + dw + 9, mid_y + 2), (30, 30, 30), -1)
            cv2.putText(frame, dist_label, (mid_x + 6, mid_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_DIST, 2, cv2.LINE_AA)

        contact_n = sum(1 for a in analysis.get("alerts", []) if a.get("level") == "CONTACT_CRITIQUE")
        proximity_n = sum(1 for a in analysis.get("alerts", []) if a.get("level") == "PROXIMITE")
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_off = 0
        for text, bg_color in (
            (f"!! CONTACT CRITIQUE: {contact_n} cas" if contact_n else "", (0, 0, 180)),
            (f"PROXIMITE DANGEREUSE: {proximity_n} cas" if proximity_n else "", (0, 110, 200)),
        ):
            if not text:
                continue
            (bw, bh), _ = cv2.getTextSize(text, font, 0.7, 2)
            pad = 8
            cv2.rectangle(frame, (0, y_off), (bw + pad * 2 + 4, y_off + bh + pad * 2), bg_color, -1)
            cv2.putText(frame, text, (pad, y_off + bh + pad), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y_off += bh + pad * 2 + 2

        return frame

    def stream_video(self, video_path: str | Path, runtime_cfg: RuntimeRuleConfig | None = None,
                     stop_event: threading.Event | None = None, command_queue: queue.Queue | None = None):
        cfg = runtime_cfg or RuntimeRuleConfig(
            danger_distance_m=self.cfg.danger_distance_default_m,
            excavator_radius_m=self.cfg.excavator_radius_default_m,
            pixel_to_meter=self.cfg.pixel_to_meter,
            speed_threshold_m_s=self.cfg.speed_threshold_m_s,
            frame_skip=self.cfg.default_skip,
        )
        stop_event = stop_event or threading.Event()
        command_queue = command_queue or queue.Queue()
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        monitor = InteractionSafetyMonitor(fps=fps, config=cfg)
        frame_number = 0
        last_dets: list[Detection] = []
        paused = False
        try:
            while cap.isOpened() and not stop_event.is_set():
                while True:
                    try:
                        cmd = command_queue.get_nowait()
                    except queue.Empty:
                        break
                    if cmd.get("type") == "pause":
                        paused = True
                    elif cmd.get("type") == "resume":
                        paused = False
                    elif cmd.get("type") == "seek":
                        target = int(cmd.get("frame", frame_number))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
                        frame_number = max(0, target)
                    elif cmd.get("type") in {"update_config", "init"}:
                        c = cmd.get("config", {})
                        circle = c.get("circle")
                        cfg = RuntimeRuleConfig(
                            danger_distance_m=float(c.get("danger_distance_m", cfg.danger_distance_m)),
                            contact_threshold_m=float(c.get("contact_threshold_m", cfg.contact_threshold_m)),
                            excavator_radius_m=float(c.get("excavator_radius_m", cfg.excavator_radius_m)),
                            pixel_to_meter=float(c.get("pixel_to_meter", cfg.pixel_to_meter)),
                            speed_threshold_m_s=float(c.get("speed_threshold_m_s", cfg.speed_threshold_m_s)),
                            auto_excavator_center=bool(c.get("auto_excavator_center", cfg.auto_excavator_center)),
                            circle_center_norm=(
                                (float(circle["cx"]), float(circle["cy"]))
                                if isinstance(circle, dict) and "cx" in circle and "cy" in circle
                                else cfg.circle_center_norm
                            ),
                            frame_skip=int(c.get("frame_skip", cfg.frame_skip)),
                        )
                        if isinstance(circle, dict) and "radius_m" in circle:
                            cfg.excavator_radius_m = float(circle["radius_m"])
                        monitor.update_config(cfg)
                if paused:
                    time.sleep(0.03)
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                frame = self._resize(frame)
                skip = max(1, int(cfg.frame_skip))
                should_detect = (
                    frame_number % skip == 0
                    or not last_dets
                    or bool(monitor.state.active_intrusions)
                )
                if should_detect:
                    last_dets = self.tracker.detect_and_track(frame)
                analysis = monitor.analyze(frame.shape[1], frame.shape[0], last_dets)
                frame = self._draw(frame, last_dets, analysis, cfg)
                ok_jpg, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok_jpg:
                    frame_number += 1
                    continue
                danger_map = {
                    pair["person_id"]: pair.get("alert_level", "PROXIMITE")
                    for pair in analysis.get("danger_pairs", [])
                }
                yield {
                    "type": "frame",
                    "frame_base64": base64.b64encode(buffer).decode("utf-8"),
                    "detections": [
                        {
                            "id": d.track_id,
                            "class": d.class_name,
                            "confidence": d.confidence,
                            "bbox": d.bbox,
                            "center": d.center,
                            "has_mask": d.mask_pts is not None,
                            "alert_level": (
                                danger_map.get(d.track_id) if d.class_name in PERSON_CLASSES else None
                            ),
                            "color_hex": _bgr_to_hex(
                                _COLOR_DANGER if danger_map.get(d.track_id) == "CONTACT_CRITIQUE"
                                else _COLOR_PROXIMITY if d.track_id in danger_map
                                else _class_color(d.class_name)
                            ),
                        }
                        for d in last_dets
                    ],
                    "alerts": analysis["alerts"],
                    "danger_pairs": analysis["danger_pairs"],
                    "stats": analysis["stats"],
                    "frame_number": frame_number,
                    "total_frames": total_frames,
                    "fps": fps,
                    "pixel_to_meter": analysis.get("pixel_to_meter"),
                }
                frame_number += 1
        finally:
            cap.release()

    @staticmethod
    def to_ws_text(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False)


_service: InteractionSafetyService | None = None


def get_service(model_path: str) -> InteractionSafetyService:
    global _service
    if _service is None:
        _service = InteractionSafetyService(ModuleConfig(model_path=model_path))
    return _service
