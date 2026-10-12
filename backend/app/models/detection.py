from datetime import datetime
from sqlalchemy import DateTime, Enum, Float, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
import enum


class EventType(str, enum.Enum):
    MISSING_HELMET = "MISSING_HELMET"
    MISSING_VEST = "MISSING_VEST"
    MISSING_GLOVES = "MISSING_GLOVES"
    MISSING_MASK = "MISSING_MASK"
    INTRUSION = "INTRUSION"
    OVERSPEED = "OVERSPEED"
    DANGEROUS_INTERACTION = "DANGEROUS_INTERACTION"
    PARKING_OCCUPIED = "PARKING_OCCUPIED"
    PARKING_FREE = "PARKING_FREE"
    CONTACT_CRITIQUE = "CONTACT_CRITIQUE"
    GENERIC = "GENERIC"


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    module_id: Mapped[int | None] = mapped_column(
        ForeignKey("ai_modules.id", ondelete="SET NULL"), nullable=True
    )
    event_type: Mapped[EventType] = mapped_column(
        Enum(EventType), default=EventType.GENERIC, nullable=False
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    frame_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    camera: Mapped["Camera"] = relationship("Camera", back_populates="detections")
    module: Mapped["AIModule"] = relationship("AIModule", back_populates="detections")
    alert: Mapped["Alert | None"] = relationship(
        "Alert", back_populates="detection", uselist=False
    )
