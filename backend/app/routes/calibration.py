"""Calibration routes for Parking and Speed modules."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.routes.ws.parking import CALIBRATION_MAP

router = APIRouter(tags=["Calibration"])


@router.post("/parking/calibrate/{video_id}")
async def parking_calibrate(video_id: str, payload: dict):
    """Save custom parking calibration for a specific video."""
    cal_path = settings.config_dir / f"calibrator_{video_id}.json"
    cal_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    CALIBRATION_MAP[video_id] = cal_path
    return {"status": "ok", "calibration_id": video_id}


@router.get("/parking/calibrate/{video_id}")
async def parking_calibrate_get(video_id: str):
    """Return saved calibration for a video."""
    cal_path = settings.config_dir / f"calibrator_{video_id}.json"
    if not cal_path.exists():
        raise HTTPException(status_code=404, detail="Calibration non trouvée")
    return json.loads(cal_path.read_text(encoding="utf-8"))


@router.post("/speed/calibrate/{video_id}")
async def speed_calibrate(video_id: str, payload: dict):
    """Save speed calibration (homography + threshold) for a specific video."""
    target = settings.config_dir / f"speed_{video_id}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"status": "ok"}


@router.get("/speed/calibrate/{video_id}")
async def speed_calibrate_get(video_id: str):
    """Return saved speed calibration for a video."""
    target = settings.config_dir / f"speed_{video_id}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Calibration non trouvée")
    return json.loads(target.read_text(encoding="utf-8"))


@router.get("/intrusion/lane-coords/{video_name}")
async def get_lane_coords(video_name: str):
    """Return saved polygon coordinates for intrusion zone."""
    coords_file = settings.config_dir / "lane_coords.json"
    if not coords_file.exists():
        return {"polygon": None}
    try:
        data = json.loads(coords_file.read_text(encoding="utf-8"))
    except Exception:
        return {"polygon": None}
    return {"polygon": data.get(video_name)}


@router.post("/intrusion/lane-coords/{video_name}")
async def save_lane_coords(video_name: str, body: dict):
    """Save polygon coordinates for intrusion zone."""
    polygon = body.get("polygon")
    if not polygon or not isinstance(polygon, list) or len(polygon) < 3:
        raise HTTPException(status_code=400, detail="Polygone invalide (minimum 3 points)")
    coords_file = settings.config_dir / "lane_coords.json"
    data: dict = {}
    if coords_file.exists():
        try:
            data = json.loads(coords_file.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[video_name] = polygon
    coords_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"saved": True, "video_name": video_name}
