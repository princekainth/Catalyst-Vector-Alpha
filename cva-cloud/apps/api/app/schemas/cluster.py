from datetime import datetime
from pydantic import BaseModel, ConfigDict


class ClusterBase(BaseModel):
    id: str
    name: str
    status: str
    agent_version: str | None
    last_seen: datetime | None
    created_at: datetime


class ClusterCreate(BaseModel):
    name: str


class ClusterOut(ClusterBase):
    model_config = ConfigDict(from_attributes=True)


class ClusterInstallResponse(BaseModel):
    cluster_id: str
    api_key: str
    install_command: str
