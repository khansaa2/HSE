"""
Karima CV Platform — Unified FastAPI Backend
Single entry point for all 5 computer vision modules on port 8000.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_tables
from app.routes import auth, cameras, modules, detections, alerts, calibration
from app.routes.video import (
    ppe_video_router,
    parking_video_router,
    intrusion_video_router,
    interaction_video_router,
    speed_video_router,
)
from app.routes.ws.ppe import router as ws_ppe_router
from app.routes.ws.parking import router as ws_parking_router
from app.routes.ws.intrusion import router as ws_intrusion_router
from app.routes.ws.speed import router as ws_speed_router
from app.routes.ws.interaction_safety import router as ws_interaction_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables, seed default AI modules and admin user
    await create_tables()
    await _seed_modules()
    await _seed_admin()
    yield
    # Shutdown (cleanup if needed)


async def _seed_modules() -> None:
    """Insert the 5 default AI modules if they don't exist yet."""
    from sqlalchemy import select
    from app.database import AsyncSessionLocal
    from app.models.ai_module import AIModule, ModuleType

    default_modules = [
        {
            "name": "PPE Detection",
            "type": ModuleType.PPE_DETECTION,
            "description": "Détection des équipements de protection individuelle (casque, gilet, gants, masque)",
        },
        {
            "name": "Intrusion Detection",
            "type": ModuleType.INTRUSION_DETECTION,
            "description": "Surveillance des zones restreintes et voies engins",
        },
        {
            "name": "Smart Parking",
            "type": ModuleType.SMART_PARKING,
            "description": "Détection d'occupation des places de parking en temps réel",
        },
        {
            "name": "Interaction Safety",
            "type": ModuleType.INTERACTION_SAFETY,
            "description": "Détection des interactions dangereuses entre personnes et engins",
        },
        {
            "name": "Vehicle Speed Estimation",
            "type": ModuleType.VEHICLE_SPEED_ESTIMATION,
            "description": "Estimation de la vitesse des véhicules par homographie",
        },
    ]

    async with AsyncSessionLocal() as db:
        for mod_data in default_modules:
            existing = await db.execute(
                select(AIModule).where(AIModule.type == mod_data["type"])
            )
            if not existing.scalar_one_or_none():
                db.add(AIModule(**mod_data, is_active=True))
        await db.commit()


async def _seed_admin() -> None:
    """Create a default admin account if no users exist yet."""
    from sqlalchemy import select, func
    from app.database import AsyncSessionLocal
    from app.models.user import User, UserRole
    from app.services.auth_service import hash_password

    async with AsyncSessionLocal() as db:
        count = await db.scalar(select(func.count()).select_from(User))
        if count == 0:
            db.add(User(
                username="admin",
                email="admin@karima.local",
                password_hash=hash_password("admin1234"),
                role=UserRole.ADMIN,
            ))
            await db.commit()


app = FastAPI(
    title="Karima CV Platform",
    description=(
        "Plateforme modulaire de vision par ordinateur pour la sécurité industrielle, "
        "le BTP et les Smart Cities."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "platform": "Karima CV Platform",
        "version": "2.0.0",
        "modules": [
            "ppe", "parking", "intrusion", "interaction-safety", "speed"
        ],
    }


# ── REST API routes ───────────────────────────────────────────────────────────
api_prefix = settings.api_prefix  # /api

app.include_router(auth.router,      prefix=api_prefix)
app.include_router(cameras.router,   prefix=api_prefix)
app.include_router(modules.router,   prefix=api_prefix)
app.include_router(detections.router, prefix=api_prefix)
app.include_router(alerts.router,    prefix=api_prefix)

# Calibration routes (no /api prefix — module-specific)
app.include_router(calibration.router)

# Per-module video management routes
app.include_router(ppe_video_router)
app.include_router(parking_video_router)
app.include_router(intrusion_video_router)
app.include_router(interaction_video_router)
app.include_router(speed_video_router)

# ── WebSocket routes ──────────────────────────────────────────────────────────
app.include_router(ws_ppe_router)
app.include_router(ws_parking_router)
app.include_router(ws_intrusion_router)
app.include_router(ws_speed_router)
app.include_router(ws_interaction_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
