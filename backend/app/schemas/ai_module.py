from datetime import datetime
from pydantic import BaseModel
from app.models.ai_module import ModuleType


class AIModuleBase(BaseModel):
    name: str
    type: ModuleType
    description: str | None = None
    is_active: bool = True


class AIModuleCreate(AIModuleBase):
    pass


class AIModuleRead(AIModuleBase):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}
