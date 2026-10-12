from datetime import datetime
from sqlalchemy import Boolean, DateTime, Enum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
import enum


class ModuleType(str, enum.Enum):
    PPE_DETECTION = "PPE_DETECTION"
    INTRUSION_DETECTION = "INTRUSION_DETECTION"
    SMART_PARKING = "SMART_PARKING"
    INTERACTION_SAFETY = "INTERACTION_SAFETY"
    VEHICLE_SPEED_ESTIMATION = "VEHICLE_SPEED_ESTIMATION"


class AIModule(Base):
    __tablename__ = "ai_modules"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    type: Mapped[ModuleType] = mapped_column(Enum(ModuleType), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    camera_modules: Mapped[list["CameraModule"]] = relationship(
        "CameraModule", back_populates="module"
    )
    detections: Mapped[list["Detection"]] = relationship(
        "Detection", back_populates="module"
    )
