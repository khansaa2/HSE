"""WebSocket route for Smart Parking."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ai_modules.parking.detector import ParkingDetector
from app.ai_modules.parking.session import ParkingTrackingSession
from app.config import settings
from app.database import AsyncSessionLocal
from app.models.ai_module import ModuleType
from app.models.detection import EventType
from app.services.detection_service import create_detection_with_alert
from app.services.module_cache import get_module_id
from app.utils.video import encode_frame, video_path as get_video_path

router = APIRouter()


def _model_path() -> Path:
    configured = Path(settings.parking_model) if settings.parking_model else None
    if configured and configured.exists():
        return configured
    old_backend = Path(__file__).resolve().parents[4] / "backend-parking-main" / "app" / "models"
    for name in ("best.onnx", "best.pt"):
        candidate = old_backend / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Modèle Parking introuvable. Vérifiez PARKING_MODEL dans .env")


def _default_calibration() -> Path:
    old_backend = Path(__file__).resolve().parents[4] / "backend-parking-main" / "app" / "config"
    return old_backend / "calibrator.json"


# Global detector instance + calibration map
_detector: ParkingDetector | None = None
_detector_lock = threading.Lock()

# video_id → source URL for calibration selection
_video_source: dict[str, str] = {}

CALIBRATION_MAP: dict[str, Path] = {}

_sessions_lock = threading.Lock()
_sessions: dict[str, ParkingTrackingSession] = {}


def _get_detector() -> ParkingDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = ParkingDetector(
                model_path=_model_path(),
                calibration_path=_default_calibration(),
            )
    return _detector


@router.websocket("/ws/parking/{video_id}")
async def ws_parking(websocket: WebSocket, video_id: str):
    await websocket.accept()

    path = get_video_path(video_id)
    if not path.exists():
        await websocket.send_json({"type": "error", "message": "Vidéo introuvable."})
        await websocket.close()
        return

    try:
        detector = _get_detector()
    except FileNotFoundError as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        return

    module_id = await get_module_id(ModuleType.SMART_PARKING)

    # Receive initial config
    spot_ids: list[str] = []
    zone_id: str = ""
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        data = json.loads(raw)
        if data.get("type") == "set_config":
            spot_ids = data.get("spot_ids", [])
            zone_id = data.get("zone_id", "")
    except Exception:
        pass

    # Select calibration
    default_cal = _default_calibration()
    source_url = _video_source.get(video_id, "")
    matched_cal = CALIBRATION_MAP.get(zone_id)
    if not (matched_cal and matched_cal.exists()):
        for pattern, cal_path in CALIBRATION_MAP.items():
            if pattern in source_url and cal_path.exists():
                matched_cal = cal_path
                break
    cal_to_use = matched_cal if matched_cal else default_cal
    if cal_to_use.exists():
        detector.load_calibration(cal_to_use)

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay = 1.0 / fps
    frame_number = 0
    spot_lookup = {s.spot_id: s for s in detector.spots}
    cal_w, cal_h = detector.calibration_size
    tracking_session = ParkingTrackingSession(spot_ids=spot_ids)
    prev_occupied: dict[str, bool] = {}

    try:
        while cap.isOpened():
            # Non-blocking mid-stream messages
            try:
                raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                msg = json.loads(raw_msg)
                if msg.get("type") == "switch_zone":
                    new_zone = msg.get("zone_id", "")
                    new_spot_ids = msg.get("spot_ids", [])
                    cal_path = CALIBRATION_MAP.get(new_zone)
                    detector.load_calibration(cal_path if cal_path and cal_path.exists() else default_cal)
                    spot_ids = new_spot_ids
                    spot_lookup = {s.spot_id: s for s in detector.spots}
                    cal_w, cal_h = detector.calibration_size
                    tracking_session = ParkingTrackingSession(spot_ids=spot_ids)
            except (asyncio.TimeoutError, Exception):
                pass

            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            if frame_number % 2 != 0:
                continue

            spots = detector.detect(frame=frame, spot_ids=spot_ids if spot_ids else None)
            now_s = frame_number / fps
            occupied_map = {s["spotId"]: bool(s.get("occupied")) for s in spots}

            for spot_id, occupied in occupied_map.items():
                prev = prev_occupied.get(spot_id)
                if prev is not None and prev != occupied:
                    event_type = EventType.PARKING_OCCUPIED if occupied else EventType.PARKING_FREE
                    try:
                        async with AsyncSessionLocal() as db:
                            await create_detection_with_alert(
                                db=db,
                                camera_id=None,
                                module_id=module_id,
                                event_type=event_type,
                                confidence=None,
                                details={"spot_id": spot_id},
                                message=f"Place {spot_id}: {'occupée' if occupied else 'libérée'}",
                            )
                            await db.commit()
                    except Exception:
                        pass
                prev_occupied[spot_id] = occupied

            tracking_meta = tracking_session.update(occupied_by_spot=occupied_map, now_s=now_s)
            for s in spots:
                sid = s.get("spotId")
                if sid is not None and sid in tracking_meta:
                    s.update(tracking_meta[sid])

            # Draw polygons
            h, w = frame.shape[:2]
            for spot in spots:
                parking_spot = spot_lookup.get(spot["spotId"])
                if parking_spot is None:
                    continue
                sx = w / float(cal_w)
                sy = h / float(cal_h)
                scaled_pts = parking_spot.polygon.copy()
                scaled_pts[:, 0] *= sx
                scaled_pts[:, 1] *= sy
                pts_np = scaled_pts.astype(np.int32)
                is_occupied = spot.get("occupied", False)
                color = (0, 0, 255) if is_occupied else (0, 255, 0)
                cv2.polylines(frame, [pts_np], True, color, 2)
                cx = int(pts_np[:, 0].mean())
                cy = int(pts_np[:, 1].mean())
                label = f"{spot['spotId']}: {'Occupé' if is_occupied else 'Libre'}"
                if is_occupied and spot["spotId"] in tracking_meta:
                    dur = int(tracking_meta[spot["spotId"]]["currentOccupiedSec"])
                    label += f" ({dur}s)"
                cv2.putText(frame, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            import base64
            frame_b64 = base64.b64encode(buf).decode("utf-8")

            occupied_count = sum(1 for s in spots if s["occupied"])
            await websocket.send_json({
                "type": "frame",
                "frame": f"data:image/jpeg;base64,{frame_b64}",
                "frame_number": frame_number,
                "fps": fps,
                "spots": spots,
                "occupiedCount": occupied_count,
                "freeCount": len(spots) - occupied_count,
            })
            await asyncio.sleep(max(0, delay * 2 - 0.01))

        await websocket.send_json({"type": "completed"})
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
