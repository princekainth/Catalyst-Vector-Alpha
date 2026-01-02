from datetime import datetime
from pydantic import BaseModel, ConfigDict


class ReasoningTraceOut(BaseModel):
    id: str
    incident_id: str
    trace_json: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
