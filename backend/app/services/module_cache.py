"""Lazy cache for AIModule id resolution (ModuleType → DB primary key)."""

from __future__ import annotations

from app.models.ai_module import ModuleType

_cache: dict[ModuleType, int | None] = {}


async def get_module_id(module_type: ModuleType) -> int | None:
    """Return the DB id for the given module type, cached after first lookup."""
    if module_type in _cache:
        return _cache[module_type]
    try:
        from sqlalchemy import select
        from app.database import AsyncSessionLocal
        from app.models.ai_module import AIModule
        async with AsyncSessionLocal() as db:
            row = await db.execute(select(AIModule.id).where(AIModule.type == module_type))
            mid = row.scalar_one_or_none()
            _cache[module_type] = mid
            return mid
    except Exception:
        _cache[module_type] = None
        return None
