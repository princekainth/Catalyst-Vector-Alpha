from datetime import datetime
from pydantic import BaseModel, ConfigDict


class ActionOut(BaseModel):
    id: str
    incident_id: str
    user_id: str | None
    action_type: str
    approved_at: datetime
    outcome: str
    notes: str

    model_config = ConfigDict(from_attributes=True)
