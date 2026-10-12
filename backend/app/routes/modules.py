from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.ai_module import AIModule
from app.schemas.ai_module import AIModuleCreate, AIModuleRead

router = APIRouter(prefix="/modules", tags=["AI Modules"])


@router.get("/", response_model=list[AIModuleRead])
async def list_modules(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AIModule).order_by(AIModule.id))
    return result.scalars().all()


@router.post("/", response_model=AIModuleRead, status_code=201)
async def create_module(payload: AIModuleCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(
        select(AIModule).where(AIModule.type == payload.type)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Un module de ce type existe déjà")
    module = AIModule(**payload.model_dump())
    db.add(module)
    await db.flush()
    await db.refresh(module)
    return module


@router.get("/{module_id}", response_model=AIModuleRead)
async def get_module(module_id: int, db: AsyncSession = Depends(get_db)):
    module = await db.get(AIModule, module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module introuvable")
    return module
