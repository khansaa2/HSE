from app.schemas.camera import CameraCreate, CameraRead, CameraUpdate
from app.schemas.ai_module import AIModuleCreate, AIModuleRead
from app.schemas.detection import DetectionCreate, DetectionRead
from app.schemas.alert import AlertRead, AlertStatusUpdate

__all__ = [
    "CameraCreate", "CameraRead", "CameraUpdate",
    "AIModuleCreate", "AIModuleRead",
    "DetectionCreate", "DetectionRead",
    "AlertRead", "AlertStatusUpdate",
]
