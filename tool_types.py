# tool_types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

# ----------------------------
# Type aliases for callbacks
# ----------------------------
ValidatorFunc = Callable[[Dict[str, Any]], bool]
NormalizerFunc = Callable[[Dict[str, Any]], Dict[str, Any]]
ToolFunc = Callable[..., Any]

# ----------------------------
# Task/role policy primitives
# ----------------------------
TASK_TYPES: Set[str] = {
    "GenericTask",
    "Observation",
    "ResourceTuning",
    "Actuation",
}

ROLE_CAPABILITIES: Dict[str, Set[str]] = {
    # map your agent "role" -> which task types it may run
    "Worker":   {"GenericTask", "ResourceTuning", "Actuation"},
    "Observer": {"GenericTask", "Observation"},
    "Security": {"GenericTask", "Observation"},
    "Planner":  {"GenericTask"},
}

# Map a tool name -> canonical task type
TOOL_TASK_TYPE: Dict[str, str] = {
    # Observation
    "get_system_cpu_load": "Observation",
    "get_system_resource_usage": "Observation",
    "top_processes": "Observation",
    "measure_responsiveness": "Observation",

    # Resource tuning
    "update_resource_allocation": "ResourceTuning",

    # Actuation (Kubernetes)
    "k8s_scale": "Actuation",
    "k8s_restart": "Actuation",

    # Information / plumbing
    "web_search": "GenericTask",
    "read_webpage": "GenericTask",
    "analyze_text_sentiment": "GenericTask",
    "create_pdf": "GenericTask",
    "generate_report_pdf": "GenericTask",
    "policy_eval": "GenericTask",
    "query_long_term_memory": "GenericTask",
    "update_world_model": "GenericTask",
    "initiate_network_scan": "GenericTask",
    "analyze_threat_signature": "GenericTask",
    "isolate_network_segment": "GenericTask",
    "kubernetes_pod_metrics": "GenericTask",
    "disk_usage": "GenericTask",
    "shuffle_roles_and_tasks": "GenericTask",
    "get_environmental_data": "GenericTask",
}

def derive_task_type(tool_name: str) -> str:
    """Return the canonical task type for a tool (default GenericTask)."""
    return TOOL_TASK_TYPE.get(tool_name, "GenericTask")

def validate_role_task_assignment(role: str, task_type: str) -> bool:
    """Check if a role is allowed to execute a given task type."""
    return task_type in ROLE_CAPABILITIES.get(role, set())

# ----------------------------
# Tool model
# ----------------------------
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]
    version: str = "1.0"
    cooldown_seconds: float = 0.0
    redact_fields: Set[str] = field(default_factory=set)
    validator: Optional[Callable[[Dict[str, Any]], bool]] = None
    normalizer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    task_type: Optional[str] = None
    roles_allowed: Set[str] = field(default_factory=set)

    _last_called_at: float = field(default=0.0, init=False, repr=False)

    def get_function_spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
    def _enforce_cooldown(self) -> Optional[str]:
        if self.cooldown_seconds <= 0: return None
        import time
        now = time.time()
        if now - self._last_called_at < self.cooldown_seconds:
            wait = max(0.0, self.cooldown_seconds - (now - self._last_called_at))
            return f"Cooldown active for '{self.name}'. Try again in {wait:.2f}s."
        self._last_called_at = now
        return None

    def call(self, **kwargs: Any) -> Any:
        """
        Thin invocation wrapper. Normalization/validation/cooldown is expected
        to be enforced by ToolRegistry.safe_call(). We still allow a local
        normalizer/validator for convenience if registry calls through here.
        """
        args = dict(kwargs)

        # Optional normalize
        if callable(self.normalizer):
            try:
                args = self.normalizer(args) or args
            except Exception:
                # If normalizer fails, just fall back to original args
                pass

        # Optional validate
        if callable(self.validator):
            if not self.validator(args):
                raise ValueError(f"Validation failed for tool '{self.name}' with args={args}")

        return self.func(**args)
