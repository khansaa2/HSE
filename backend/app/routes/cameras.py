from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.camera import Camera
from app.schemas.camera import CameraCreate, CameraRead, CameraUpdate

router = APIRouter(prefix="/cameras", tags=["Cameras"])


@router.get("/", response_model=list[CameraRead])
async def list_cameras(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Camera).order_by(Camera.id))
    return result.scalars().all()


@router.post("/", response_model=CameraRead, status_code=201)
async def create_camera(payload: CameraCreate, db: AsyncSession = Depends(get_db)):
    camera = Camera(**payload.model_dump())
    db.add(camera)
    await db.flush()
    await db.refresh(camera)
    return camera


@router.get("/{camera_id}", response_model=CameraRead)
async def get_camera(camera_id: int, db: AsyncSession = Depends(get_db)):
    camera = await db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Caméra introuvable")
    return camera


@router.put("/{camera_id}", response_model=CameraRead)
async def update_camera(
    camera_id: int, payload: CameraUpdate, db: AsyncSession = Depends(get_db)
):
    camera = await db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Caméra introuvable")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(camera, field, value)
    await db.flush()
    await db.refresh(camera)
    return camera


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: int, db: AsyncSession = Depends(get_db)):
    camera = await db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Caméra introuvable")
    await db.delete(camera)
