"""WebSocket route for Vehicle Speed Estimation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ai_modules.speed_estimation.detector import SpeedEstimator, get_model
from app.config import settings
from app.database import AsyncSessionLocal
from app.models.ai_module import ModuleType
from app.models.detection import EventType
from app.services.detection_service import create_detection_with_alert
from app.services.module_cache import get_module_id
from app.utils.video import video_path as get_video_path

router = APIRouter()


def _model_path() -> Path:
    configured = Path(settings.speed_model) if settings.speed_model else None
    if configured and configured.exists():
        return configured
    old_backend = Path(__file__).resolve().parents[4] / "backend-speed-main" / "app" / "models"
    for name in ("best.onnx", "best.pt"):
        candidate = old_backend / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Modèle Speed introuvable. Vérifiez SPEED_MODEL dans .env")


@router.websocket("/ws/speed/{video_id}")
async def ws_speed(websocket: WebSocket, video_id: str):
    await websocket.accept()

    path = get_video_path(video_id)
    if not path.exists():
        await websocket.send_json({"type": "error", "message": "Vidéo introuvable."})
        await websocket.close()
        return

    try:
        model = get_model(_model_path())
    except FileNotFoundError as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        return

    # Default calibration
    image_points = [[520.0, 680.0], [1390.0, 680.0], [1750.0, 1040.0], [180.0, 1040.0]]
    world_points = [[0.0, 0.0], [20.0, 0.0], [20.0, 8.0], [0.0, 8.0]]
    speed_threshold_kmh = 20.0
    frame_stride = 1
    stream_sleep_ms = 0

    # Load per-video calibration
    cal_file = settings.config_dir / f"speed_{video_id}.json"
    if cal_file.exists():
        try:
            cfg = json.loads(cal_file.read_text(encoding="utf-8"))
            image_points = cfg.get("image_points", image_points)
            world_points = cfg.get("world_points", world_points)
            speed_threshold_kmh = float(cfg.get("speed_threshold_kmh", speed_threshold_kmh))
            frame_stride = int(cfg.get("frame_stride", frame_stride) or frame_stride)
            stream_sleep_ms = int(cfg.get("stream_sleep_ms", stream_sleep_ms) or stream_sleep_ms)
        except Exception:
            pass

    # Receive client config
    try:
        msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        data = json.loads(msg)
        if data.get("type") == "set_config":
            image_points = data.get("image_points", image_points)
            world_points = data.get("world_points", world_points)
            speed_threshold_kmh = float(data.get("speed_threshold_kmh", speed_threshold_kmh))
            frame_stride = int(data.get("frame_stride", frame_stride) or frame_stride)
            stream_sleep_ms = int(data.get("stream_sleep_ms", stream_sleep_ms) or stream_sleep_ms)
    except Exception:
        pass

    module_id = await get_module_id(ModuleType.VEHICLE_SPEED_ESTIMATION)

    estimator = SpeedEstimator(model=model)
    h_matrix = estimator.compute_homography(image_points=image_points, world_points=world_points)

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0
    frame_stride = max(1, min(frame_stride, 10))

    frame_number = 0
    names_cache: dict[int, str] | None = None
    allowed_ids: set[int] | None = None
    last_ts_s: float | None = None
    overspeed_alerted: set[int] = set()

    import base64

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            if frame_number % frame_stride != 0:
                continue

            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            now_s = float(ts_ms) / 1000.0 if ts_ms and ts_ms > 0 else (frame_number / fps)
            if last_ts_s is not None and now_s <= last_ts_s:
                now_s = last_ts_s + (frame_stride / fps)
            last_ts_s = now_s

            pts = np.array(image_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 180, 0), 2)

            result = model.track(
                source=frame, persist=True, tracker="bytetrack.yaml",
                conf=0.3, iou=0.5, verbose=False,
            )[0]

            if names_cache is None:
                names_cache = {int(k): str(v) for k, v in result.names.items()}
                allowed_ids = estimator.allowed_classes(names_cache)

            tracks_payload: list[dict] = []
            if result.boxes is not None and result.boxes.id is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                tids = result.boxes.id.int().cpu().tolist()
                cids = result.boxes.cls.int().cpu().tolist()
                for box, tid, cid in zip(xyxy, tids, cids):
                    if allowed_ids is not None and cid not in allowed_ids:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    cx, cy = estimator.ground_point(box)
                    world = estimator.px_to_world(h_matrix, cx, cy)
                    speed = estimator.update_speed(tid, world, now_s)

                    state = estimator.track_state[tid]
                    state["trajectory"].append((int(cx), int(cy)))
                    tr = list(state["trajectory"])
                    for i in range(1, len(tr)):
                        cv2.line(frame, tr[i - 1], tr[i], (80, 220, 80), 2)

                    color = (0, 255, 0) if speed < speed_threshold_kmh else (0, 0, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"ID {tid} | {names_cache.get(cid, str(cid))} | {speed:.1f} km/h"
                    if speed >= speed_threshold_kmh:
                        label += " | OVERSPEED"
                        if tid not in overspeed_alerted:
                            overspeed_alerted.add(tid)
                            try:
                                async with AsyncSessionLocal() as db:
                                    await create_detection_with_alert(
                                        db=db,
                                        camera_id=None,
                                        module_id=module_id,
                                        event_type=EventType.OVERSPEED,
                                        confidence=None,
                                        details={
                                            "track_id": tid,
                                            "class": names_cache.get(cid, str(cid)),
                                            "speed_kmh": round(speed, 2),
                                            "threshold_kmh": speed_threshold_kmh,
                                        },
                                        message=f"Vitesse excessive: {speed:.1f} km/h (seuil: {speed_threshold_kmh} km/h)",
                                    )
                                    await db.commit()
                            except Exception:
                                pass
                    cv2.putText(frame, label, (int(x1), max(20, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                    tracks_payload.append({
                        "id": tid,
                        "class": names_cache.get(cid, str(cid)),
                        "speedKmh": round(speed, 2),
                        "overspeed": speed >= speed_threshold_kmh,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    })

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            frame_b64 = base64.b64encode(buf).decode("utf-8")
            await websocket.send_json({
                "type": "frame",
                "frame": f"data:image/jpeg;base64,{frame_b64}",
                "frame_number": frame_number,
                "fps": fps,
                "tracks": tracks_payload,
            })
            if stream_sleep_ms > 0:
                await asyncio.sleep(stream_sleep_ms / 1000.0)

        await websocket.send_json({"type": "completed"})
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
