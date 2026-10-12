# Karima CV Platform — Backend Unifié

Plateforme modulaire de vision par ordinateur pour la sécurité industrielle, le BTP et les Smart Cities.

## Architecture

```
backend/
├── app/
│   ├── main.py                  ← FastAPI app (port 8000)
│   ├── config.py                ← Settings (pydantic-settings)
│   ├── database.py              ← SQLAlchemy async + PostgreSQL
│   │
│   ├── models/                  ← ORM SQLAlchemy
│   │   ├── user.py
│   │   ├── camera.py
│   │   ├── ai_module.py
│   │   ├── camera_module.py
│   │   ├── detection.py
│   │   └── alert.py
│   │
│   ├── schemas/                 ← Pydantic schemas (validation)
│   │
│   ├── routes/
│   │   ├── cameras.py           ← GET/POST/PUT/DELETE /api/cameras
│   │   ├── modules.py           ← GET/POST /api/modules
│   │   ├── detections.py        ← GET /api/detections
│   │   ├── alerts.py            ← GET /api/alerts, PUT /api/alerts/{id}/status
│   │   ├── video.py             ← Routes vidéo par module (/ppe/, /parking/, etc.)
│   │   ├── calibration.py       ← Calibration parking + speed
│   │   └── ws/                  ← WebSocket routes
│   │       ├── ppe.py           ← ws://localhost:8000/ws/ppe/{video_id}
│   │       ├── parking.py       ← ws://localhost:8000/ws/parking/{video_id}
│   │       ├── intrusion.py     ← ws://localhost:8000/ws/intrusion/{video_id}
│   │       ├── speed.py         ← ws://localhost:8000/ws/speed/{video_id}
│   │       └── interaction_safety.py ← ws://localhost:8000/ws/interaction-safety/{video_id}
│   │
│   ├── ai_modules/              ← Logique IA (migrée depuis anciens backends)
│   │   ├── ppe/detector.py
│   │   ├── parking/detector.py + session.py
│   │   ├── intrusion/detector.py
│   │   ├── speed_estimation/detector.py
│   │   └── interaction_safety/
│   │       ├── constants.py
│   │       ├── tracker.py
│   │       ├── rules.py
│   │       └── service.py
│   │
│   ├── services/
│   │   └── detection_service.py ← Création auto alertes depuis détections
│   │
│   └── utils/
│       ├── video.py             ← Utilitaires vidéo partagés
│       └── geometry.py          ← Point-in-polygon, etc.
│
├── uploads/                     ← Vidéos uploadées/téléchargées
├── config/                      ← Calibrations JSON (parking, speed, zones)
├── cache/                       ← Cache détections
├── models/                      ← Fichiers modèles YOLO (.pt / .onnx)
├── requirements.txt
└── .env
```

## Base de données PostgreSQL

### Tables

| Table | Description |
|-------|-------------|
| `users` | Utilisateurs de la plateforme |
| `cameras` | Caméras (CAMERA) ou vidéos temporaires (VIDEO) |
| `ai_modules` | Les 5 modules IA disponibles |
| `camera_modules` | Relation many-to-many camera ↔ module |
| `detections` | Événements détectés par les modules |
| `alerts` | Alertes générées automatiquement depuis les détections |

### Types

**SourceType:** `VIDEO` | `CAMERA`
**ModuleType:** `PPE_DETECTION` | `INTRUSION_DETECTION` | `SMART_PARKING` | `INTERACTION_SAFETY` | `VEHICLE_SPEED_ESTIMATION`
**EventType:** `MISSING_HELMET` | `INTRUSION` | `OVERSPEED` | `DANGEROUS_INTERACTION` | `PARKING_OCCUPIED` | ...
**AlertSeverity:** `LOW` | `MEDIUM` | `HIGH` | `CRITICAL`
**AlertStatus:** `NEW` | `ACKNOWLEDGED` | `RESOLVED`

## API REST

```
GET    /api/cameras
POST   /api/cameras
GET    /api/cameras/{id}
PUT    /api/cameras/{id}
DELETE /api/cameras/{id}

GET    /api/modules
POST   /api/modules

GET    /api/detections?camera_id=&module_id=&limit=

GET    /api/alerts?status=&severity=&limit=
PUT    /api/alerts/{id}/status
```

## Routes WebSocket

| Module | URL |
|--------|-----|
| PPE Detection | `ws://localhost:8000/ws/ppe/{video_id}` |
| Smart Parking | `ws://localhost:8000/ws/parking/{video_id}` |
| Intrusion / Voie Engins | `ws://localhost:8000/ws/intrusion/{video_id}` |
| Interaction Safety | `ws://localhost:8000/ws/interaction-safety/{video_id}` |
| Vehicle Speed | `ws://localhost:8000/ws/speed/{video_id}` |

## Gestion Vidéo par Module

Chaque module expose les mêmes routes de gestion vidéo :

```
GET  /{module}/test-videos
POST /{module}/load-test-video/{video_name}
POST /{module}/load_url
POST /{module}/upload
```

**Exemples :**
```
GET  /ppe/test-videos
POST /parking/load_url
POST /speed/upload
```

## Calibration

```
POST /parking/calibrate/{video_id}   ← Sauvegarder calibration places parking
GET  /parking/calibrate/{video_id}

POST /speed/calibrate/{video_id}     ← Sauvegarder calibration homographie
GET  /speed/calibrate/{video_id}

POST /intrusion/lane-coords/{video_name}
GET  /intrusion/lane-coords/{video_name}
```

## Installation et Démarrage

### 1. Prérequis

- Python 3.11+
- PostgreSQL 15+
- Les anciens backends (pour les modèles YOLO et vidéos de test)

### 2. PostgreSQL

```sql
-- Créer la base de données et l'utilisateur
CREATE USER karima WITH PASSWORD 'karima';
CREATE DATABASE karima_cv OWNER karima;
GRANT ALL PRIVILEGES ON DATABASE karima_cv TO karima;
```

### 3. Installation Python

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Configuration

Éditer `backend/.env` :

```env
DATABASE_URL=postgresql+asyncpg://karima:karima@localhost:5432/karima_cv

# Optionnel : chemins vers les modèles YOLO
# Laisser vide = auto-détection depuis les anciens backends
PPE_MODEL=
PARKING_MODEL=
INTRUSION_MODEL=
INTERACTION_MODEL=
SPEED_MODEL=
```

### 5. Démarrage

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Le backend démarre sur **http://localhost:8000**

- Documentation API : http://localhost:8000/docs
- Health check : http://localhost:8000/health

### 6. Frontend

```bash
cd gitex2026
npm install
npm run dev
```

Le frontend démarre sur **http://localhost:5173**

## Modèles YOLO

Le backend détecte automatiquement les modèles dans les anciens répertoires.
Pour les centraliser dans `backend/models/` :

```bash
# Copier les modèles (adapter les chemins)
cp backend-epi-main/app/models/best.pt          backend/models/ppe.pt
cp backend-parking-main/app/models/best.pt      backend/models/parking.pt
cp backend-zones-dangereuses/app/models/bestm.pt backend/models/intrusion.pt
cp backend-interaction-safety/app/models/bestse.pt backend/models/interaction.pt
cp backend-speed-main/app/models/best.pt        backend/models/speed.pt
```

Puis mettre à jour `backend/.env` :

```env
PPE_MODEL=models/ppe.pt
PARKING_MODEL=models/parking.pt
INTRUSION_MODEL=models/intrusion.pt
INTERACTION_MODEL=models/interaction.pt
SPEED_MODEL=models/speed.pt
```

## Architecture Caméras

```
Camera (VIDEO ou CAMERA)
  └── CameraModule (enabled/disabled)
        └── AIModule (type)

Exemples :
  Caméra Zone Production (VIDEO: video/test_ppe.mp4)
    → PPE Detection
    → Intrusion Detection

  Caméra Zone Parking (CAMERA: rtsp://camera-ip/stream)
    → Smart Parking

  Caméra Entrée
    → Vehicle Speed Estimation
    → Interaction Safety
```

## Migration depuis les anciens backends

Les anciens backends (backend-epi-main, backend-parking-main, etc.) peuvent continuer
à fonctionner en parallèle pendant la migration.

Le nouveau backend unifié pointe automatiquement vers :
- Les **modèles YOLO** dans les anciens répertoires (auto-détection)
- Les **vidéos de test** dans les anciens répertoires `test_videos/`
- Les **calibrations** sauvegardées dans `backend/config/`

Aucune copie de fichiers n'est nécessaire pour démarrer.
