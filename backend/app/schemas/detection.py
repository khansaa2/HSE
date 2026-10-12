from datetime import datetime
from pydantic import BaseModel
from app.models.detection import EventType


class DetectionCreate(BaseModel):
    camera_id: int | None = None
    module_id: int | None = None
    event_type: EventType = EventType.GENERIC
    confidence: float | None = None
    frame_path: str | None = None
    details: dict | None = None


class DetectionRead(DetectionCreate):
    id: int
    detected_at: datetime

    model_config = {"from_attributes": True}
