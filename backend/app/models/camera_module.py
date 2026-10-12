from sqlalchemy import Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class CameraModule(Base):
    """Many-to-many: Camera <-> AIModule with extra config."""

    __tablename__ = "camera_modules"
    __table_args__ = (UniqueConstraint("camera_id", "module_id", name="uq_camera_module"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )
    module_id: Mapped[int] = mapped_column(
        ForeignKey("ai_modules.id", ondelete="CASCADE"), nullable=False
    )
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    camera: Mapped["Camera"] = relationship("Camera", back_populates="camera_modules")
    module: Mapped["AIModule"] = relationship("AIModule", back_populates="camera_modules")
