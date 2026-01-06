from datetime import datetime
from pydantic import BaseModel, ConfigDict


class IncidentBase(BaseModel):
    id: str
    cluster_id: str
    namespace: str
    pod_name: str
    issue_type: str
    severity: str
    status: str
    summary: str
    action_type: str | None = None
    action_config: str | None = None
    outcome: str | None = None
    executed_at: datetime | None = None
    completed_at: datetime | None = None
    archived_at: datetime | None = None
    created_at: datetime


class IncidentOut(IncidentBase):
    model_config = ConfigDict(from_attributes=True)


class IncidentFilter(BaseModel):
    cluster_id: str | None = None
    status: str | None = None
    severity: str | None = None
    issue_type: str | None = None


class IncidentReport(BaseModel):
    cluster_id: str
    pod_name: str
    namespace: str
    issue_type: str
    severity: str
    status: str = "pending"
    summary: str | None = None
    action_type: str | None = None
    action_config: dict | None = None
    reasoning_trace: dict


class IncidentUpdate(BaseModel):
    status: str
    outcome: dict | None = None
    action_type: str | None = None
