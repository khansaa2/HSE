from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.alert import Alert
from app.schemas.alert import AlertRead, AlertStatusUpdate

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("/", response_model=list[AlertRead])
async def list_alerts(
    status: str | None = Query(None),
    severity: str | None = Query(None),
    limit: int = Query(100, le=500),
    db: AsyncSession = Depends(get_db),
):
    query = select(Alert).order_by(Alert.created_at.desc()).limit(limit)
    if status:
        query = query.where(Alert.status == status)
    if severity:
        query = query.where(Alert.severity == severity)
    result = await db.execute(query)
    return result.scalars().all()


@router.put("/{alert_id}/status", response_model=AlertRead)
async def update_alert_status(
    alert_id: int,
    payload: AlertStatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    alert = await db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alerte introuvable")
    alert.status = payload.status
    await db.flush()
    await db.refresh(alert)
    return alert
