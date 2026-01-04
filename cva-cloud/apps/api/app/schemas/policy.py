from datetime import datetime
from pydantic import BaseModel, ConfigDict


class PolicyBase(BaseModel):
    id: str
    org_id: str
    cluster_id: str | None
    name: str
    issue_type: str
    auto_approve: bool
    max_memory_mb: int | None
    allow_placeholder: bool
    status: str
    created_at: datetime
    updated_at: datetime


class PolicyOut(PolicyBase):
    model_config = ConfigDict(from_attributes=True)


class PolicyCreate(BaseModel):
    name: str
    issue_type: str
    cluster_id: str | None = None
    auto_approve: bool = False
    max_memory_mb: int | None = None
    allow_placeholder: bool = False


class PolicyUpdate(BaseModel):
    auto_approve: bool | None = None
    max_memory_mb: int | None = None
    allow_placeholder: bool | None = None
    status: str | None = None
