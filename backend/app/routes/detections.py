from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.detection import Detection
from app.schemas.detection import DetectionRead

router = APIRouter(prefix="/detections", tags=["Detections"])


@router.get("/", response_model=list[DetectionRead])
async def list_detections(
    camera_id: int | None = Query(None),
    module_id: int | None = Query(None),
    limit: int = Query(100, le=500),
    db: AsyncSession = Depends(get_db),
):
    query = select(Detection).order_by(Detection.detected_at.desc()).limit(limit)
    if camera_id is not None:
        query = query.where(Detection.camera_id == camera_id)
    if module_id is not None:
        query = query.where(Detection.module_id == module_id)
    result = await db.execute(query)
    return result.scalars().all()
