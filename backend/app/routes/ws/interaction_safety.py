"""WebSocket route for Interaction Safety."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Query, WebSocket

from app.ai_modules.interaction_safety.rules import RuntimeRuleConfig
from app.ai_modules.interaction_safety.service import get_service
from app.config import settings
from app.database import AsyncSessionLocal
from app.models.ai_module import ModuleType
from app.models.detection import EventType
from app.services.detection_service import create_detection_with_alert
from app.services.module_cache import get_module_id
from app.utils.video import video_path as get_video_path

_INTERACTION_COOLDOWN = 10.0

router = APIRouter()


def _model_path() -> str:
    configured = settings.interaction_model
    if configured and Path(configured).exists():
        return configured
    old_backend = Path(__file__).resolve().parents[4] / "backend-interaction-safety" / "app" / "models"
    for name in ("bestse.pt", "best.onnx", "best.pt"):
        candidate = old_backend / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Modèle Interaction Safety introuvable. Vérifiez INTERACTION_MODEL dans .env")


@router.websocket("/ws/interaction-safety/{video_id}")
async def ws_interaction_safety(
    websocket: WebSocket,
    video_id: str,
    danger_distance_m: float = Query(default=2.5),
    excavator_radius_m: float = Query(default=3.0),
    contact_threshold_m: float = Query(default=0.3),
    pixel_to_meter: float = Query(default=0.02),
    frame_skip: int = Query(default=2),
):
    await websocket.accept()

    path = get_video_path(video_id)
    if not path.exists():
        await websocket.send_text(json.dumps({"type": "error", "message": "Vidéo introuvable."}))
        await websocket.close()
        return

    try:
        model_path = _model_path()
    except FileNotFoundError as e:
        await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        await websocket.close()
        return

    module_id = await get_module_id(ModuleType.INTERACTION_SAFETY)

    stop_event = threading.Event()
    command_queue: queue.Queue = queue.Queue()
    runtime_cfg = RuntimeRuleConfig(
        danger_distance_m=danger_distance_m,
        excavator_radius_m=excavator_radius_m,
        contact_threshold_m=contact_threshold_m,
        pixel_to_meter=pixel_to_meter,
        frame_skip=frame_skip,
    )

    async def reader():
        while True:
            try:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                if msg.get("type") in {"pause", "resume", "seek", "update_config", "init"}:
                    command_queue.put(msg)
            except Exception:
                break

    reader_task = asyncio.create_task(reader())

    last_contact_save: float = -9999.0
    last_proximity_save: float = -9999.0

    try:
        service = get_service(model_path)
        for payload in service.stream_video(
            video_path=path,
            runtime_cfg=runtime_cfg,
            stop_event=stop_event,
            command_queue=command_queue,
        ):
            alerts = payload.get("alerts", [])
            if alerts:
                now = time.monotonic()
                has_contact = any(a.get("level") == "CONTACT_CRITIQUE" for a in alerts)
                has_proximity = any(a.get("level") == "PROXIMITE" for a in alerts)

                if has_contact and now - last_contact_save >= _INTERACTION_COOLDOWN:
                    last_contact_save = now
                    try:
                        async with AsyncSessionLocal() as db:
                            await create_detection_with_alert(
                                db=db,
                                camera_id=None,
                                module_id=module_id,
                                event_type=EventType.CONTACT_CRITIQUE,
                                confidence=None,
                                details={"alerts": [a for a in alerts if a.get("level") == "CONTACT_CRITIQUE"]},
                                message="Contact critique personne-engin détecté",
                            )
                            await db.commit()
                    except Exception:
                        pass

                if has_proximity and now - last_proximity_save >= _INTERACTION_COOLDOWN:
                    last_proximity_save = now
                    try:
                        async with AsyncSessionLocal() as db:
                            await create_detection_with_alert(
                                db=db,
                                camera_id=None,
                                module_id=module_id,
                                event_type=EventType.DANGEROUS_INTERACTION,
                                confidence=None,
                                details={"alerts": [a for a in alerts if a.get("level") == "PROXIMITE"]},
                                message="Interaction dangereuse personne-engin détectée",
                            )
                            await db.commit()
                    except Exception:
                        pass

            await websocket.send_text(service.to_ws_text(payload))
            await asyncio.sleep(0)
        await websocket.send_text(json.dumps({"type": "completed", "message": "Analyse terminée"}))
    except Exception:
        pass
    finally:
        stop_event.set()
        reader_task.cancel()


@router.websocket("/ws/interaction_safety/{video_id}")
async def ws_interaction_safety_alt(
    websocket: WebSocket,
    video_id: str,
    danger_distance_m: float = Query(default=2.5),
    excavator_radius_m: float = Query(default=3.0),
    contact_threshold_m: float = Query(default=0.3),
    pixel_to_meter: float = Query(default=0.02),
    frame_skip: int = Query(default=2),
):
    """Alias with underscore for backward compatibility."""
    await ws_interaction_safety(
        websocket, video_id, danger_distance_m, excavator_radius_m,
        contact_threshold_m, pixel_to_meter, frame_skip,
    )
