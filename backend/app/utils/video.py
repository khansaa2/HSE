"""Shared video utilities used by all AI modules."""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import uuid
from pathlib import Path

import cv2
import httpx
import numpy as np

from app.config import settings


def video_id_from_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:32]


def video_id_from_name(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:32]


def video_path(video_id: str) -> Path:
    return settings.uploads_dir / f"{video_id}.mp4"


def video_metadata(path: Path | str) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, first = cap.read()
    cap.release()

    first_b64 = ""
    if ret:
        _, buf = cv2.imencode(".jpg", first, [cv2.IMWRITE_JPEG_QUALITY, 85])
        first_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "first_frame": first_b64,
    }


async def download_video(url: str) -> tuple[str, Path]:
    """Download video from URL, return (video_id, path). Skips if already cached."""
    vid_id = video_id_from_url(url)
    path = video_path(vid_id)
    if not path.exists():
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            path.write_bytes(resp.content)
    return vid_id, path


def load_test_video(video_name: str, test_dir: Path) -> tuple[str, Path]:
    """Link a test video into uploads/ and return (video_id, session_path)."""
    if ".." in video_name or "/" in video_name or "\\" in video_name:
        raise ValueError("Nom de vidéo invalide")
    if not video_name.endswith(".mp4"):
        video_name += ".mp4"

    source = test_dir / video_name
    if not source.exists():
        raise FileNotFoundError(f"Vidéo '{video_name}' introuvable dans {test_dir}")

    vid_id = video_id_from_name(video_name)
    session_path = video_path(vid_id)
    if not session_path.exists():
        try:
            os.symlink(str(source.resolve()), str(session_path))
        except OSError:
            shutil.copy2(source, session_path)

    return vid_id, session_path


async def save_uploaded_video(content: bytes) -> tuple[str, Path]:
    vid_id = str(uuid.uuid4())
    path = video_path(vid_id)
    path.write_bytes(content)
    return vid_id, path


def resize_frame(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def encode_frame(frame: np.ndarray, quality: int = 72) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")
