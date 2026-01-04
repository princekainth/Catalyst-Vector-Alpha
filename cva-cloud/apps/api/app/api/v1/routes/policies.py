from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_org_id
from app.db.session import get_db
from app.models.policy import Policy
from app.schemas.policy import PolicyCreate, PolicyOut, PolicyUpdate

router = APIRouter(prefix="/policies", tags=["policies"])


@router.get("/", response_model=list[PolicyOut])
def list_policies(
    org_id: str = Depends(get_org_id),
    db: Session = Depends(get_db),
):
    return db.query(Policy).filter(Policy.org_id == org_id).all()


@router.post("/", response_model=PolicyOut)
def create_policy(
    payload: PolicyCreate,
    org_id: str = Depends(get_org_id),
    db: Session = Depends(get_db),
):
    policy = Policy(
        id=f"pol-{uuid.uuid4()}",
        org_id=org_id,
        cluster_id=payload.cluster_id,
        name=payload.name,
        issue_type=payload.issue_type,
        auto_approve=payload.auto_approve,
        max_memory_mb=payload.max_memory_mb,
        allow_placeholder=payload.allow_placeholder,
        status="active",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(policy)
    db.commit()
    db.refresh(policy)
    return policy


@router.patch("/{policy_id}", response_model=PolicyOut)
def update_policy(
    policy_id: str,
    payload: PolicyUpdate,
    org_id: str = Depends(get_org_id),
    db: Session = Depends(get_db),
):
    policy = db.query(Policy).filter(Policy.id == policy_id, Policy.org_id == org_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    if payload.auto_approve is not None:
        policy.auto_approve = payload.auto_approve
    if payload.max_memory_mb is not None:
        policy.max_memory_mb = payload.max_memory_mb
    if payload.allow_placeholder is not None:
        policy.allow_placeholder = payload.allow_placeholder
    if payload.status is not None:
        policy.status = payload.status
    policy.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(policy)
    return policy


@router.delete("/{policy_id}")
def delete_policy(
    policy_id: str,
    org_id: str = Depends(get_org_id),
    db: Session = Depends(get_db),
):
    policy = db.query(Policy).filter(Policy.id == policy_id, Policy.org_id == org_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    db.delete(policy)
    db.commit()
    return {"status": "deleted"}
