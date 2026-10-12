from datetime import datetime
from sqlalchemy import DateTime, Enum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
import enum


class SourceType(str, enum.Enum):
    VIDEO = "VIDEO"
    CAMERA = "CAMERA"


class CameraStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MAINTENANCE = "MAINTENANCE"


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    location: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source_type: Mapped[SourceType] = mapped_column(
        Enum(SourceType), default=SourceType.VIDEO, nullable=False
    )
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[CameraStatus] = mapped_column(
        Enum(CameraStatus), default=CameraStatus.ACTIVE, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    camera_modules: Mapped[list["CameraModule"]] = relationship(
        "CameraModule", back_populates="camera", cascade="all, delete-orphan"
    )
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="camera"
    )
