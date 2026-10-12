from datetime import datetime
from pydantic import BaseModel
from app.models.camera import CameraStatus, SourceType


class CameraBase(BaseModel):
    name: str
    location: str | None = None
    source_type: SourceType = SourceType.VIDEO
    source_url: str | None = None
    status: CameraStatus = CameraStatus.ACTIVE


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):
    name: str | None = None
    location: str | None = None
    source_type: SourceType | None = None
    source_url: str | None = None
    status: CameraStatus | None = None


class CameraRead(CameraBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}
