"""WebSocket route for PPE Detection."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ai_modules.ppe.detector import get_detector
from app.config import settings
from app.database import AsyncSessionLocal
from app.models.ai_module import ModuleType
from app.models.detection import EventType
from app.services.detection_service import create_detection_with_alert
from app.services.module_cache import get_module_id
from app.utils.video import encode_frame, resize_frame, video_path as get_video_path

_PPE_EVENT_MAP: dict[str, EventType] = {
    "no helmet": EventType.MISSING_HELMET,
    "no vest": EventType.MISSING_VEST,
    "no gloves": EventType.MISSING_GLOVES,
    "no mask": EventType.MISSING_MASK,
}

router = APIRouter()

# Model path: look for the PPE model in the configured path or fall back to old backend
def _model_path() -> Path:
    configured = Path(settings.ppe_model) if settings.ppe_model else None
    if configured and configured.exists():
        return configured
    # Fall back to old backend location
    old_backend = Path(__file__).resolve().parents[4] / "backend-epi-main" / "app" / "models"
    for name in ("best.onnx", "best.pt"):
        candidate = old_backend / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Modèle PPE introuvable. Vérifiez PPE_MODEL dans .env")


@router.websocket("/ws/ppe/{video_id}")
async def ws_ppe(websocket: WebSocket, video_id: str):
    await websocket.accept()

    path = get_video_path(video_id)
    if not path.exists():
        await websocket.send_json({"type": "error", "message": "Vidéo introuvable."})
        await websocket.close()
        return

    try:
        detector = get_detector(_model_path())
    except FileNotFoundError as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        return

    module_id = await get_module_id(ModuleType.PPE_DETECTION)

    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_duration = 1.0 / fps
    frame_number = 0
    last_detections: list[dict] = []
    alerted_violations: set[tuple[int, str]] = set()

    try:
        while cap.isOpened():
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            frame = resize_frame(frame, settings.max_frame_width)

            if frame_number % 2 == 0 or not last_detections:
                loop = asyncio.get_event_loop()
                last_detections = await loop.run_in_executor(
                    None,
                    lambda f=frame.copy(): detector.track(f),
                )
                for det in last_detections:
                    if not det.get("violation"):
                        continue
                    tid = det.get("track_id", -1)
                    cls = det.get("class", "").lower()
                    event_type = next(
                        (et for prefix, et in _PPE_EVENT_MAP.items() if cls.startswith(prefix)),
                        None,
                    )
                    if event_type is None:
                        continue
                    key = (tid, event_type.value)
                    if key in alerted_violations:
                        continue
                    alerted_violations.add(key)
                    try:
                        async with AsyncSessionLocal() as db:
                            await create_detection_with_alert(
                                db=db,
                                camera_id=None,
                                module_id=module_id,
                                event_type=event_type,
                                confidence=det.get("confidence"),
                                details={"track_id": tid, "class": cls, "bbox": det.get("bbox")},
                                message=f"EPI manquant: {cls} (track #{tid})",
                            )
                            await db.commit()
                    except Exception:
                        pass

            await websocket.send_json({
                "type": "frame",
                "frame": encode_frame(frame, settings.jpeg_quality),
                "detections": last_detections,
                "frame_number": frame_number,
                "total_frames": total_frames,
                "fps": fps,
            })

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, frame_duration - elapsed))

        await websocket.send_json({"type": "completed", "message": "Analyse terminée."})
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
