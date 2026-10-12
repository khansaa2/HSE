from datetime import datetime
from pydantic import BaseModel
from app.models.alert import AlertSeverity, AlertStatus


class AlertRead(BaseModel):
    id: int
    detection_id: int
    title: str
    message: str | None
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertStatusUpdate(BaseModel):
    status: AlertStatus
