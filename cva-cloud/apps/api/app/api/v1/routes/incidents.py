import json
from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, Query, Header, HTTPException, Body
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
    query = db.query(Incident).filter(Incident.archived_at.is_(None))
    if status:
        query = query.filter(Incident.status == status)
    if severity:
        query = query.filter(Incident.severity == severity)
    if issue_type:
        query = query.filter(Incident.issue_type == issue_type)
    if cluster_id:
        query = query.filter(Incident.cluster_id == cluster_id)
    return query.all()


@router.get("/archived", response_model=list[IncidentOut])
@router.get("/archived/", response_model=list[IncidentOut])
def list_archived_incidents(
    db: Session = Depends(get_db),
    cluster_id: str | None = Query(default=None),
):
    query = db.query(Incident).filter(Incident.archived_at.is_not(None))
    if cluster_id:
        query = query.filter(Incident.cluster_id == cluster_id)
    return query.order_by(Incident.archived_at.desc()).all()


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
            Incident.namespace == payload.namespace,
            Incident.pod_name == payload.pod_name,
            Incident.issue_type == payload.issue_type,
            Incident.status.in_(["pending", "open"]),
            Incident.archived_at.is_(None),
        )
        .order_by(Incident.created_at.desc())
        .first()
    )
    if existing:
        updated = False
        if payload.action_type and not existing.action_type:
            existing.action_type = payload.action_type
            updated = True
        if payload.action_config and (not existing.action_config or existing.action_config in ("", "{}")):
            existing.action_config = json.dumps(payload.action_config)
            updated = True
        if updated:
            db.commit()
        return {"incident_id": existing.id}
    incident = Incident(
        id=str(uuid.uuid4()),
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
    db.flush()
    trace = ReasoningTrace(
        id=f"trace-{uuid.uuid4()}",
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
    incident = (
        db.query(Incident)
        .filter(
            Incident.id == incident_id,
            Incident.archived_at.is_(None),
        )
        .first()
    )
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident


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
            Incident.archived_at.is_(None),
        )
        .all()
    )


@router.post("/{incident_id}/restore")
def restore_incident(
    incident_id: str,
    db: Session = Depends(get_db),
):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.archived_at = None
    db.commit()
    return {"status": "restored", "incident_id": incident_id}


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
    if payload.action_type:
        incident.action_type = payload.action_type
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
    action_config: dict = {}
    if incident.action_config:
        try:
            action_config = json.loads(incident.action_config)
        except Exception:
            action_config = {}
    if not incident.action_type:
        recs = action_config.get("recommended_actions") or []
        if recs and isinstance(recs[0], dict):
            incident.action_type = recs[0].get("action") or incident.action_type
            action_config.update(recs[0])
        if not incident.action_type:
            incident.action_type = "generic_remediation"
    action_config.setdefault("pod_name", incident.pod_name)
    action_config.setdefault("namespace", incident.namespace)
    action_config.setdefault("issue_type", incident.issue_type)
    action_config.setdefault("message", incident.summary)
    incident.action_config = json.dumps(action_config)
    incident.status = "approved"
    incident.executed_at = None
    incident.completed_at = None
    incident.outcome = ""
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


@router.post("/history/clear")
def clear_history(payload: dict | None = Body(default=None), db: Session = Depends(get_db)):
    ids = None
    if payload:
        ids = payload.get("ids")
    if ids is not None and len(ids) == 0:
        return {"deleted": 0}
    if ids:
        query = db.query(Incident).filter(Incident.id.in_(ids))
    else:
        query = db.query(Incident).filter(Incident.status.in_(["dismissed", "fixed", "failed"]))
    incident_ids = [row[0] for row in query.with_entities(Incident.id).all()]
    deleted = 0
    if incident_ids:
        deleted = (
            db.query(Incident)
            .filter(Incident.id.in_(incident_ids))
            .update({Incident.archived_at: datetime.utcnow()}, synchronize_session=False)
        )
    db.commit()
    return {"deleted": deleted}


@router.get("/export")
@router.get("/export/")
def export_incidents(
    db: Session = Depends(get_db),
    scope: str = Query(default="archived"),
):
    if scope == "all":
        incidents = db.query(Incident).all()
    else:
        incidents = db.query(Incident).filter(Incident.archived_at.is_not(None)).all()
    incident_ids = [incident.id for incident in incidents]
    traces = []
    if incident_ids:
        traces = (
            db.query(ReasoningTrace)
            .filter(ReasoningTrace.incident_id.in_(incident_ids))
            .all()
        )
    return {
        "scope": scope,
        "count": len(incidents),
        "incidents": incidents,
        "traces": traces,
    }
