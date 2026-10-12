"""WebSocket route for Intrusion / Voie Engins detection."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ai_modules.intrusion.detector import (
    DEFAULT_POLYGON,
    IntrusionDetector,
    MachineLaneMonitor,
    annotate_frame,
)
from app.config import settings
from app.database import AsyncSessionLocal
from app.models.ai_module import ModuleType
from app.models.detection import EventType
from app.services.detection_service import create_detection_with_alert
from app.services.module_cache import get_module_id
from app.utils.video import resize_frame, video_path as get_video_path

_INTRUSION_COOLDOWN = 10.0

router = APIRouter()

_detector: IntrusionDetector | None = None


def _model_path() -> Path:
    configured = Path(settings.intrusion_model) if settings.intrusion_model else None
    if configured and configured.exists():
        return configured
    old_backend = Path(__file__).resolve().parents[4] / "backend-zones-dangereuses" / "app" / "models"
    for name in ("bestm.pt", "best.onnx", "best.pt"):
        candidate = old_backend / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Modèle Intrusion introuvable. Vérifiez INTRUSION_MODEL dans .env")


def _get_detector() -> IntrusionDetector:
    global _detector
    if _detector is None:
        _detector = IntrusionDetector(_model_path())
    return _detector


@router.websocket("/ws/intrusion/{video_id}")
async def ws_intrusion(websocket: WebSocket, video_id: str):
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

    module_id = await get_module_id(ModuleType.INTRUSION_DETECTION)

    # Wait for polygon config
    polygon = DEFAULT_POLYGON
    show_polygon = True
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
        data = json.loads(raw)
        if data.get("type") == "set_polygon" and data.get("polygon"):
            polygon = data["polygon"]
        show_polygon = data.get("show_polygon", True)
    except Exception:
        pass

    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_duration = 1.0 / fps
    frame_number = 0
    last_detections: list[dict] = []
    monitor = MachineLaneMonitor(polygon_points=polygon)
    last_intrusion_save: float = -9999.0

    # Queue for incoming client messages
    incoming: asyncio.Queue = asyncio.Queue()

    async def _read_client():
        while True:
            try:
                text = await websocket.receive_text()
                await incoming.put(json.loads(text))
            except Exception:
                break

    reader_task = asyncio.create_task(_read_client())

    try:
        while cap.isOpened():
            t_start = time.monotonic()

            # Process incoming messages
            while not incoming.empty():
                msg = incoming.get_nowait()
                if msg.get("type") == "set_polygon" and msg.get("polygon"):
                    polygon = msg["polygon"]
                    monitor = MachineLaneMonitor(polygon_points=polygon)
                elif msg.get("type") == "toggle_polygon":
                    show_polygon = msg.get("visible", show_polygon)

            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            frame = resize_frame(frame, settings.max_frame_width)

            if frame_number % 2 == 0 or not last_detections:
                loop = asyncio.get_event_loop()
                last_detections = await loop.run_in_executor(
                    None, lambda f=frame.copy(): detector.detect(f)
                )

            result = monitor.analyze(frame.shape, last_detections)

            if result.get("person_in_lane"):
                now = time.monotonic()
                if now - last_intrusion_save >= _INTRUSION_COOLDOWN:
                    last_intrusion_save = now
                    try:
                        async with AsyncSessionLocal() as db:
                            await create_detection_with_alert(
                                db=db,
                                camera_id=None,
                                module_id=module_id,
                                event_type=EventType.INTRUSION,
                                confidence=None,
                                details={"frame_number": frame_number},
                                message="Intrusion détectée dans la zone restreinte",
                            )
                            await db.commit()
                    except Exception:
                        pass

            payload = annotate_frame(frame, last_detections, result, frame_number, total_frames, fps, show_polygon)
            await websocket.send_text(json.dumps(payload))

            elapsed = time.monotonic() - t_start
            await asyncio.sleep(max(0.0, frame_duration - elapsed))

        await websocket.send_json({"type": "completed", "message": "Analyse terminée."})
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()
        cap.release()
