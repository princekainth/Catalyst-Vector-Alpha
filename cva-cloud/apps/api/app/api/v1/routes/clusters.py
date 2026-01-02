from datetime import datetime
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from app.core.security import get_org_id, verify_token
from app.db.session import get_db
from app.models.cluster import Cluster
from app.models.incident import Incident
from app.schemas.cluster import ClusterOut, ClusterCreate, ClusterInstallResponse

router = APIRouter(prefix="/clusters", tags=["clusters"])


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


@router.get("/", response_model=list[ClusterOut])
def list_clusters(
    user_id: str = Depends(verify_token),
    org_id: str = Depends(get_org_id),
    db: Session = Depends(get_db),
):
    return (
        db.query(Cluster)
        .filter(Cluster.org_id == org_id, Cluster.user_id == user_id)
        .all()
    )


@router.get("/{cluster_id}", response_model=ClusterOut)
@router.get("/{cluster_id}/", response_model=ClusterOut)
def get_cluster(
    cluster_id: str,
    user_id: str = Depends(verify_token),
    org_id: str = Depends(get_org_id),
    db: Session = Depends(get_db),
):
    cluster = (
        db.query(Cluster)
        .filter(
            Cluster.id == cluster_id,
            Cluster.org_id == org_id,
            Cluster.user_id == user_id,
        )
        .first()
    )
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    return cluster


@router.post("/", response_model=ClusterInstallResponse)
def create_cluster(
    payload: ClusterCreate,
    user_id: str = Depends(verify_token),
    org_id: str = Depends(get_org_id),
    db: Session = Depends(get_db),
):
    existing = db.query(Cluster).filter(Cluster.org_id == org_id).count()
    if existing >= 10:
        raise HTTPException(status_code=429, detail="Cluster limit reached")

    api_key = str(uuid.uuid4())
    cluster_id = f"cluster-{uuid.uuid4()}"
    cluster = Cluster(
        id=cluster_id,
        org_id=org_id,
        user_id=user_id,
        name=payload.name,
        api_key=api_key,
        status="pending",
        agent_version=None,
        last_seen=None,
        created_at=datetime.utcnow(),
    )
    db.add(cluster)
    db.commit()
    db.refresh(cluster)
    install_command = f"kubectl apply -f https://cva.yourdomain.com/install/{cluster_id}/{api_key}"
    return {
        "cluster_id": cluster_id,
        "api_key": api_key,
        "install_command": install_command,
    }


@router.post("/{cluster_id}/heartbeat")
@router.post("/{cluster_id}/heartbeat/")
def cluster_heartbeat(
    cluster_id: str,
    payload: dict,
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
):
    cluster = _get_cluster_by_api_key(db, cluster_id, authorization)

    cluster.status = "connected"
    cluster.last_seen = datetime.utcnow()
    cluster.agent_version = payload.get("agent_version") or cluster.agent_version
    db.commit()
    return {"status": "ok"}


@router.get("/{cluster_id}/pending-actions")
@router.get("/{cluster_id}/pending-actions/")
def get_pending_actions(
    cluster_id: str,
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
):
    cluster = _get_cluster_by_api_key(db, cluster_id, authorization)
    pending = (
        db.query(Incident)
        .filter(
            Incident.cluster_id == cluster.id,
            Incident.status == "approved",
            Incident.executed_at.is_(None),
        )
        .all()
    )
    actions = []
    now = datetime.utcnow()
    for incident in pending:
        action_config = {}
        if incident.action_config:
            try:
                action_config = json.loads(incident.action_config)
            except Exception:
                action_config = {}
        action_config.update({
            "pod_name": incident.pod_name,
            "namespace": incident.namespace,
            "issue_type": incident.issue_type,
            "message": incident.summary,
        })
        actions.append(
            {
                "incident_id": incident.id,
                "action_type": incident.action_type or "generic_remediation",
                "action_config": action_config,
            }
        )
        incident.executed_at = now
    db.commit()
    return {"actions": actions}
