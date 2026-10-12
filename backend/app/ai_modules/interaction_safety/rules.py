"""Business rules for Interaction Safety — distance calculation and alert levels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .constants import AT_RISK_CLASSES, MACHINE_CLASSES
from .tracker import Detection

_PERSON_HEIGHT_M: float = 1.75
_CALIB_MIN_HEIGHT_PX: float = 30.0
_CALIB_EMA_ALPHA: float = 0.3


def _euclidean(p1: list[float], p2: list[float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


@dataclass
class RuntimeRuleConfig:
    danger_distance_m: float = 2.5
    contact_threshold_m: float = 0.3
    excavator_radius_m: float = 3.0
    pixel_to_meter: float = 0.02
    speed_threshold_m_s: float = 0.15
    auto_excavator_center: bool = True
    circle_center_norm: tuple[float, float] | None = None
    frame_skip: int = 2
    homography_pts: Optional[list[list[float]]] = None
    homography_real_dim: Optional[list[float]] = None


@dataclass
class InteractionState:
    intrusions_total: int = 0
    max_duration_s: float = 0.0
    active_intrusions: dict[str, float] = field(default_factory=dict)
    previous_centers: dict[int, list[float]] = field(default_factory=dict)


def _build_homography_matrix(pts_norm, real_w_m, real_h_m, frame_w, frame_h):
    try:
        import cv2
        if len(pts_norm) != 4:
            return None
        src = np.array([[p[0] * frame_w, p[1] * frame_h] for p in pts_norm], dtype=np.float32)
        dst = np.array([[0.0, 0.0], [real_w_m, 0.0], [real_w_m, real_h_m], [0.0, real_h_m]], dtype=np.float32)
        M, _ = cv2.findHomography(src, dst)
        return M
    except Exception:
        return None


def _transform_point(M, px, py):
    import cv2
    pt = np.array([[[px, py]]], dtype=np.float32)
    t = cv2.perspectiveTransform(pt, M)
    return float(t[0][0][0]), float(t[0][0][1])


def _distance_m_with_homography(M, p1_px, p2_px):
    m1 = _transform_point(M, p1_px[0], p1_px[1])
    m2 = _transform_point(M, p2_px[0], p2_px[1])
    return math.sqrt((m1[0] - m2[0]) ** 2 + (m1[1] - m2[1]) ** 2)


class InteractionSafetyMonitor:
    def __init__(self, fps: float, config: RuntimeRuleConfig | None = None):
        self.fps = fps if fps > 0 else 25.0
        self.cfg = config or RuntimeRuleConfig()
        self.state = InteractionState()
        self._H: np.ndarray | None = None
        self._last_frame_size: tuple[int, int] = (0, 0)
        self._eff_p2m: float = self.cfg.pixel_to_meter

    def update_config(self, new_cfg: RuntimeRuleConfig) -> None:
        self.cfg = new_cfg
        self._H = None

    def _ensure_homography(self, frame_w, frame_h):
        if (not self.cfg.homography_pts or not self.cfg.homography_real_dim
                or len(self.cfg.homography_pts) != 4 or len(self.cfg.homography_real_dim) != 2):
            self._H = None
            return
        if self._H is not None and self._last_frame_size == (frame_w, frame_h):
            return
        real_w, real_h = self.cfg.homography_real_dim
        if real_w <= 0 or real_h <= 0:
            self._H = None
            return
        self._H = _build_homography_matrix(self.cfg.homography_pts, real_w, real_h, frame_w, frame_h)
        self._last_frame_size = (frame_w, frame_h)

    def _machine_speed_m_s(self, machine: Detection, frame_w: int, frame_h: int) -> float:
        prev = self.state.previous_centers.get(machine.track_id)
        self.state.previous_centers[machine.track_id] = machine.center
        if prev is None:
            return 0.0
        if self._H is not None:
            delta_m = _distance_m_with_homography(self._H, prev, machine.center)
        else:
            delta_m = _euclidean(prev, machine.center) * self._eff_p2m
        return delta_m * self.fps

    def _distance_meters(self, p1_px, p2_px):
        if self._H is not None:
            return _distance_m_with_homography(self._H, p1_px, p2_px)
        return _euclidean(p1_px, p2_px) * self._eff_p2m

    @staticmethod
    def _person_feet(person: Detection) -> list[float]:
        if person.mask_pts and len(person.mask_pts) >= 3:
            bottom = max(person.mask_pts, key=lambda p: p[1])
            return [float(bottom[0]), float(bottom[1])]
        px1, _, px2, py2 = person.bbox
        return [(px1 + px2) / 2.0, float(py2)]

    @staticmethod
    def _nearest_on_machine(feet: list[float], machine: Detection) -> list[float]:
        if machine.mask_pts and len(machine.mask_pts) >= 3:
            fx, fy = feet[0], feet[1]
            best_pt: list[float] = [0.0, 0.0]
            best_dist_sq = float("inf")
            pts = machine.mask_pts
            n = len(pts)
            for i in range(n):
                ax, ay = pts[i][0], pts[i][1]
                bx, by = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
                dx, dy = bx - ax, by - ay
                len_sq = dx * dx + dy * dy
                if len_sq == 0:
                    nx, ny = ax, ay
                else:
                    t = max(0.0, min(1.0, ((fx - ax) * dx + (fy - ay) * dy) / len_sq))
                    nx, ny = ax + t * dx, ay + t * dy
                dist_sq = (nx - fx) ** 2 + (ny - fy) ** 2
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_pt = [nx, ny]
            return best_pt
        mx1, my1, mx2, my2 = machine.bbox
        return [max(mx1, min(feet[0], mx2)), max(my1, min(feet[1], my2))]

    def _measure_person_machine(self, person: Detection, machine: Detection):
        feet = self._person_feet(person)
        if machine.mask_pts and len(machine.mask_pts) >= 3:
            import cv2
            arr = np.array(machine.mask_pts, dtype=np.float32)
            if cv2.pointPolygonTest(arr, (float(feet[0]), float(feet[1])), False) >= 0:
                return 0.0, feet, feet
        nearest = self._nearest_on_machine(feet, machine)
        dist_m = self._distance_meters(feet, nearest)
        return dist_m, feet, nearest

    def analyze(self, frame_w: int, frame_h: int, detections: list[Detection]) -> dict:
        self._ensure_homography(frame_w, frame_h)
        persons = [d for d in detections if d.class_name in AT_RISK_CLASSES]
        machines = [d for d in detections if d.class_name in MACHINE_CLASSES]

        if persons and self._H is None:
            heights = []
            for p in persons:
                if p.mask_pts and len(p.mask_pts) >= 3:
                    ys = [pt[1] for pt in p.mask_pts]
                    heights.append(max(ys) - min(ys))
                else:
                    heights.append(p.bbox[3] - p.bbox[1])
            tallest_h = max(heights)
            if tallest_h >= _CALIB_MIN_HEIGHT_PX:
                est = _PERSON_HEIGHT_M / tallest_h
                self._eff_p2m = _CALIB_EMA_ALPHA * est + (1.0 - _CALIB_EMA_ALPHA) * self._eff_p2m

        alerts: list[dict] = []
        danger_pairs: list[dict] = []
        frame_intrusion_keys: set[str] = set()

        auto_excavator_center_px = None
        excavators = [m for m in machines if m.class_name == "excavator"]
        if self.cfg.auto_excavator_center and excavators:
            best_exc = max(excavators, key=lambda d: d.confidence)
            if best_exc.mask_pts and len(best_exc.mask_pts) >= 3:
                cx = sum(p[0] for p in best_exc.mask_pts) / len(best_exc.mask_pts)
                cy = sum(p[1] for p in best_exc.mask_pts) / len(best_exc.mask_pts)
                auto_excavator_center_px = (cx, cy)
            else:
                auto_excavator_center_px = (best_exc.center[0], best_exc.center[1])

        for machine in machines:
            speed_m_s = self._machine_speed_m_s(machine, frame_w, frame_h)
            threshold_m = self.cfg.danger_distance_m
            for person in persons:
                distance_m, feet, nearest = self._measure_person_machine(person, machine)
                if machine.class_name == "excavator":
                    if self.cfg.circle_center_norm:
                        cx_n, cy_n = self.cfg.circle_center_norm
                        center_px = [cx_n * frame_w, cy_n * frame_h]
                    elif auto_excavator_center_px:
                        center_px = list(auto_excavator_center_px)
                    else:
                        center_px = machine.center
                    distance_m = self._distance_meters(feet, center_px)
                    threshold_m = self.cfg.excavator_radius_m
                    nearest = center_px
                if distance_m > threshold_m:
                    continue
                alert_level = (
                    "CONTACT_CRITIQUE" if distance_m <= self.cfg.contact_threshold_m
                    else "PROXIMITE"
                )
                key = f"{person.track_id}:{machine.track_id}"
                frame_intrusion_keys.add(key)
                if key not in self.state.active_intrusions:
                    self.state.active_intrusions[key] = 0.0
                    self.state.intrusions_total += 1
                    prefix = "Contact critique" if alert_level == "CONTACT_CRITIQUE" else "Proximite dangereuse"
                    alerts.append({
                        "level": alert_level,
                        "type": "interaction",
                        "message": f"{prefix}: personne #{person.track_id} / {machine.class_name} #{machine.track_id}",
                        "person_id": person.track_id,
                        "machine_id": machine.track_id,
                        "machine_type": machine.class_name,
                    })
                self.state.active_intrusions[key] += 1.0 / self.fps
                self.state.max_duration_s = max(self.state.max_duration_s, self.state.active_intrusions[key])
                danger_pairs.append({
                    "person_id": person.track_id,
                    "machine_id": machine.track_id,
                    "machine_type": machine.class_name,
                    "distance_m": round(distance_m, 2),
                    "threshold_m": round(threshold_m, 2),
                    "contact_threshold_m": round(self.cfg.contact_threshold_m, 2),
                    "alert_level": alert_level,
                    "speed_m_s": round(speed_m_s, 2),
                    "duration_s": round(self.state.active_intrusions[key], 2),
                    "person_feet": [round(feet[0], 1), round(feet[1], 1)],
                    "machine_nearest_pt": [round(nearest[0], 1), round(nearest[1], 1)],
                })

        stale = [k for k in self.state.active_intrusions if k not in frame_intrusion_keys]
        for k in stale:
            del self.state.active_intrusions[k]

        return {
            "alerts": alerts,
            "danger_pairs": danger_pairs,
            "pixel_to_meter": round(self._eff_p2m, 5),
            "stats": {
                "intrusions_total": self.state.intrusions_total,
                "active_intrusions": len(self.state.active_intrusions),
                "max_duration_s": round(self.state.max_duration_s, 2),
            },
        }
