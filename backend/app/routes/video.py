"""Shared video management routes — common to all modules."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.utils.video import (
    download_video,
    load_test_video,
    save_uploaded_video,
    video_metadata,
    video_path,
)
from pydantic import BaseModel

router = APIRouter(tags=["Video Management"])


class LoadUrlRequest(BaseModel):
    url: str


def _test_videos_dir(module: str) -> Path:
    """Return the test_videos directory for a given module."""
    old_backends = {
        "ppe": "backend-epi-main",
        "parking": "backend-parking-main",
        "intrusion": "backend-zones-dangereuses",
        "interaction-safety": "backend-interaction-safety",
        "speed": "backend-speed-main",
    }
    backend_dir = old_backends.get(module, f"backend-{module}-main")
    return Path(__file__).resolve().parents[3] / backend_dir / "app" / "test_videos"


def _make_video_routes(prefix: str, module_key: str) -> APIRouter:
    sub = APIRouter(prefix=prefix)

    @sub.get("/test-videos")
    async def list_test_videos():
        test_dir = _test_videos_dir(module_key)
        if not test_dir.exists():
            return {"videos": []}
        videos = [{"fileName": vf.name, "label": vf.stem} for vf in test_dir.glob("*.mp4")]
        return {"videos": videos}

    @sub.post("/load-test-video/{video_name}")
    async def load_test_video_route(video_name: str):
        test_dir = _test_videos_dir(module_key)
        try:
            vid_id, path = load_test_video(video_name, test_dir)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=404, detail=str(e))
        meta = video_metadata(path)
        return {"video_id": vid_id, **meta}

    @sub.post("/load_url")
    async def load_url(request: LoadUrlRequest):
        try:
            vid_id, path = await download_video(request.url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur téléchargement: {e}")
        meta = video_metadata(path)
        return {"video_id": vid_id, **meta}

    @sub.post("/upload")
    async def upload_video(file: UploadFile = File(...)):
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Fichier vide")
        vid_id, path = await save_uploaded_video(content)
        meta = video_metadata(path)
        return {"video_id": vid_id, **meta}

    return sub


# Create per-module video routers
ppe_video_router = _make_video_routes("/ppe", "ppe")
parking_video_router = _make_video_routes("/parking", "parking")
intrusion_video_router = _make_video_routes("/intrusion", "intrusion")
interaction_video_router = _make_video_routes("/interaction-safety", "interaction-safety")
speed_video_router = _make_video_routes("/speed", "speed")
