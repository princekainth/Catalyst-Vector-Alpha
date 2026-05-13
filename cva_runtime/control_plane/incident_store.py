import os
import json
import uuid
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field

@dataclass
class Incident:
    id: str
    incident_type: str
    severity: str
    namespace: str
    workload: str
    pod: Optional[str] = None
    status: str = "OPEN" # OPEN, GATED, APPROVED, REMEDIATING, RESOLVED, FAILED, MANUAL
    evidence: str = ""
    classification: str = ""
    recommended_tool: Optional[str] = None
    recommended_args_redacted: Optional[str] = None
    risk: str = "SAFE"
    trace_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    audit_refs: List[str] = field(default_factory=list)
    transitions: List[Dict[str, str]] = field(default_factory=list)

class IncidentStore:
    def __init__(self, persistence_file: str = ".cva/incidents/incidents.jsonl"):
        self.persistence_file = persistence_file
        self.incidents: Dict[str, Incident] = {}
        self.lock = threading.Lock()
        self._ensure_dir()
        self._load_from_disk()

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)

    def _load_from_disk(self):
        if not os.path.exists(self.persistence_file):
            return
        with self.lock:
            try:
                with open(self.persistence_file, "r") as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        if data.get("type") == "INCIDENT_SNAPSHOT":
                            inc_data = data["incident"]
                            # Clean up data for dataclass compat
                            valid_keys = Incident.__dataclass_fields__.keys()
                            filtered_data = {k: v for k, v in inc_data.items() if k in valid_keys}
                            self.incidents[inc_data["id"]] = Incident(**filtered_data)
            except Exception as e:
                print(f"Error loading incidents: {e}")

    def _persist(self, incident: Incident, event_type: str = "INCIDENT_UPDATE"):
        try:
            with open(self.persistence_file, "a") as f:
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "INCIDENT_SNAPSHOT",
                    "event": event_type,
                    "incident": asdict(incident)
                }
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"Error persisting incident: {e}")

    def create_or_update_incident(self, **kwargs) -> str:
        with self.lock:
            # Check for existing open incident with same type/target
            ns = kwargs.get("namespace", "unknown")
            wl = kwargs.get("workload", "unknown")
            it = kwargs.get("incident_type", "unknown")
            
            existing_id = None
            for inc in self.incidents.values():
                if inc.status not in ["RESOLVED", "FAILED"] and \
                   inc.namespace == ns and inc.workload == wl and inc.incident_type == it:
                    existing_id = inc.id
                    break
            
            if existing_id:
                incident = self.incidents[existing_id]
                # Update fields
                for k, v in kwargs.items():
                    if k == "status":
                        self._update_status_locked(incident, v)
                    elif k == "evidence" and v:
                        # Cap evidence
                        val = str(v)
                        if len(val) > 4000:
                            val = val[:3997] + "..."
                        incident.evidence = val
                    elif hasattr(incident, k) and v is not None:
                        setattr(incident, k, v)
                incident.updated_at = datetime.now(timezone.utc).isoformat()
                self._persist(incident, "INCIDENT_UPDATE")
                return incident.id
            else:
                inc_id = f"inc_{uuid.uuid4().hex[:8]}"
                # Ensure id is in kwargs for dataclass
                kwargs["id"] = inc_id
                # Cap evidence if present
                if "evidence" in kwargs and kwargs["evidence"]:
                    val = str(kwargs["evidence"])
                    if len(val) > 4000:
                        kwargs["evidence"] = val[:3997] + "..."
                
                # Filter unknown kwargs
                valid_keys = Incident.__dataclass_fields__.keys()
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
                
                incident = Incident(**filtered_kwargs)
                incident.transitions.append({
                    "from": "NONE", "to": "OPEN", "timestamp": incident.created_at
                })
                self.incidents[inc_id] = incident
                self._persist(incident, "INCIDENT_CREATE")
                return inc_id

    def attach_trace(self, incident_id: str, trace_id: str, tool: str = None, risk: str = None):
        with self.lock:
            if incident_id in self.incidents:
                incident = self.incidents[incident_id]
                incident.trace_id = trace_id
                if tool: incident.recommended_tool = tool
                if risk: incident.risk = risk
                if incident.status == "OPEN":
                    self._update_status_locked(incident, "GATED")
                incident.updated_at = datetime.now(timezone.utc).isoformat()
                self._persist(incident, "TRACE_ATTACHED")

    def _update_status_locked(self, incident: Incident, new_status: str):
        if incident.status != new_status:
            incident.transitions.append({
                "from": incident.status,
                "to": new_status,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            incident.status = new_status
            incident.updated_at = datetime.now(timezone.utc).isoformat()

    def update_status(self, incident_id: str, status: str):
        with self.lock:
            if incident_id in self.incidents:
                incident = self.incidents[incident_id]
                self._update_status_locked(incident, status)
                self._persist(incident, "STATUS_UPDATE")

    def resolve_incident(self, incident_id: str, reason: str = "manual"):
        with self.lock:
            if incident_id in self.incidents:
                incident = self.incidents[incident_id]
                self._update_status_locked(incident, "RESOLVED")
                incident.resolved_at = datetime.now(timezone.utc).isoformat()
                incident.evidence += f"\n[Resolution] {reason}"
                self._persist(incident, "INCIDENT_RESOLVE")

    def list_incidents(self, status: str = None) -> List[Dict]:
        with self.lock:
            res = [asdict(i) for i in self.incidents.values()]
            if status:
                res = [r for r in res if r["status"] == status]
            return sorted(res, key=lambda x: x["updated_at"], reverse=True)

    def get_incident(self, incident_id: str) -> Optional[Dict]:
        with self.lock:
            inc = self.incidents.get(incident_id)
            return asdict(inc) if inc else None

# Global Singleton
incident_store = IncidentStore()
