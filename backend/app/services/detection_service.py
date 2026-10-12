"""Service for persisting detections and auto-generating alerts."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.alert import Alert, AlertSeverity, AlertStatus
from app.models.detection import Detection, EventType


# Maps event types to alert severity
_SEVERITY_MAP: dict[EventType, AlertSeverity] = {
    EventType.MISSING_HELMET: AlertSeverity.HIGH,
    EventType.MISSING_VEST: AlertSeverity.MEDIUM,
    EventType.MISSING_GLOVES: AlertSeverity.MEDIUM,
    EventType.MISSING_MASK: AlertSeverity.MEDIUM,
    EventType.INTRUSION: AlertSeverity.CRITICAL,
    EventType.OVERSPEED: AlertSeverity.HIGH,
    EventType.DANGEROUS_INTERACTION: AlertSeverity.CRITICAL,
    EventType.CONTACT_CRITIQUE: AlertSeverity.CRITICAL,
    EventType.PARKING_OCCUPIED: AlertSeverity.LOW,
    EventType.PARKING_FREE: AlertSeverity.LOW,
    EventType.GENERIC: AlertSeverity.LOW,
}

_TITLE_MAP: dict[EventType, str] = {
    EventType.MISSING_HELMET: "Casque manquant détecté",
    EventType.MISSING_VEST: "Gilet de sécurité manquant",
    EventType.MISSING_GLOVES: "Gants manquants",
    EventType.MISSING_MASK: "Masque manquant",
    EventType.INTRUSION: "Intrusion dans zone restreinte",
    EventType.OVERSPEED: "Vitesse excessive détectée",
    EventType.DANGEROUS_INTERACTION: "Interaction dangereuse personne-engin",
    EventType.CONTACT_CRITIQUE: "Contact critique personne-engin",
    EventType.PARKING_OCCUPIED: "Place de parking occupée",
    EventType.PARKING_FREE: "Place de parking libérée",
    EventType.GENERIC: "Événement détecté",
}


async def create_detection_with_alert(
    db: AsyncSession,
    camera_id: int | None,
    module_id: int | None,
    event_type: EventType,
    confidence: float | None = None,
    frame_path: str | None = None,
    details: dict | None = None,
    message: str | None = None,
) -> tuple[Detection, Alert]:
    """Create a detection record and automatically generate an alert."""
    detection = Detection(
        camera_id=camera_id,
        module_id=module_id,
        event_type=event_type,
        confidence=confidence,
        frame_path=frame_path,
        details=details,
    )
    db.add(detection)
    await db.flush()

    severity = _SEVERITY_MAP.get(event_type, AlertSeverity.MEDIUM)
    title = _TITLE_MAP.get(event_type, "Événement détecté")

    alert = Alert(
        detection_id=detection.id,
        title=title,
        message=message or f"Détecté à la caméra #{camera_id}, module #{module_id}",
        severity=severity,
        status=AlertStatus.NEW,
    )
    db.add(alert)
    await db.flush()

    return detection, alert
