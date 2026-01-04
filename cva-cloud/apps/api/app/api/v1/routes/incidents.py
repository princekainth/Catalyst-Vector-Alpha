import json
from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, Query, Header, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_org_id, verify_token
from app.db.session import get_db
from app.models.cluster import Cluster
from app.models.incident import Incident
from app.schemas.incident import IncidentOut, IncidentReport, IncidentUpdate
from app.schemas.reasoning import ReasoningTraceOut
from app.models.reasoning_trace import ReasoningTrace

router = APIRouter(prefix="/incidents", tags=["incidents"])


def _get_cluster_by_api_key(
    db: Session, cluster_id: str, authorization: str | None
) -> Cluster:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")
    cluster = (
        db.query(Cluster)
        .filter(Cluster.id == cluster_id, Cluster.api_key == token)
        .first()
    )
    if not cluster:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return cluster


@router.get("/", response_model=list[IncidentOut])
def list_incidents(
    db: Session = Depends(get_db),
    status: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    issue_type: str | None = Query(default=None),
    cluster_id: str | None = Query(default=None),
):
    query = db.query(Incident)
    if status:
        query = query.filter(Incident.status == status)
    if severity:
        query = query.filter(Incident.severity == severity)
    if issue_type:
        query = query.filter(Incident.issue_type == issue_type)
    if cluster_id:
        query = query.filter(Incident.cluster_id == cluster_id)
    return query.all()


@router.post("/report")
def report_incident(
    payload: IncidentReport,
    db: Session = Depends(get_db),
):
    cluster = (
        db.query(Cluster)
        .filter(Cluster.id == payload.cluster_id)
        .first()
    )
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    existing = (
        db.query(Incident)
        .filter(
            Incident.cluster_id == cluster.id,
            Incident.pod_name == payload.pod_name,
            Incident.issue_type == payload.issue_type,
            Incident.status == "pending",
        )
        .first()
    )
    if existing:
        return {"incident_id": existing.id}
    incident = Incident(
        id=f"inc-{uuid.uuid4()}",
        cluster_id=cluster.id,
        user_id=cluster.user_id,
        namespace=payload.namespace,
        pod_name=payload.pod_name,
        issue_type=payload.issue_type,
        severity=payload.severity,
        status=payload.status,
        summary=payload.summary or payload.issue_type,
        action_type=payload.action_type or "",
        action_config=json.dumps(payload.action_config or {}),
        created_at=datetime.utcnow(),
    )
    db.add(incident)
    trace = ReasoningTrace(
        id=f"trace-{incident.id}",
        incident_id=incident.id,
        trace_json=json.dumps(payload.reasoning_trace),
        created_at=datetime.utcnow(),
    )
    db.add(trace)
    db.commit()
    return {"incident_id": incident.id}


@router.get("/{incident_id}", response_model=IncidentOut)
@router.get("/{incident_id}/", response_model=IncidentOut)
def get_incident(
    incident_id: str,
    db: Session = Depends(get_db),
):
    return (
        db.query(Incident)
        .filter(
            Incident.id == incident_id,
        )
        .first()
    )


@router.get("/{incident_id}/trace", response_model=list[ReasoningTraceOut])
@router.get("/{incident_id}/trace/", response_model=list[ReasoningTraceOut])
def get_reasoning_trace(
    incident_id: str,
    db: Session = Depends(get_db),
):
    return (
        db.query(ReasoningTrace)
        .join(Incident)
        .filter(
            Incident.id == incident_id,
        )
        .all()
    )


@router.patch("/{incident_id}")
@router.patch("/{incident_id}/")
def update_incident_status(
    incident_id: str,
    payload: IncidentUpdate,
    db: Session = Depends(get_db),
):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.status = payload.status
    if payload.outcome is not None:
        incident.outcome = json.dumps(payload.outcome)
    incident.completed_at = datetime.utcnow()
    db.commit()
    return {"status": "ok"}


@router.post("/{incident_id}/approve")
def approve_incident(
    incident_id: str,
    db: Session = Depends(get_db),
):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.status = "approved"
    db.commit()
    return {"status": "approved", "incident_id": incident_id}


@router.post("/{incident_id}/rollback")
def rollback_incident(
    incident_id: str,
    db: Session = Depends(get_db),
):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.status = "rollback_requested"
    db.commit()
    return {"status": "rollback_requested", "incident_id": incident_id}
