# tool_registry.py  
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import os
import time
import logging
from typing import Optional, Dict, List, Any, Tuple, Set
from urllib.parse import urlparse

def _is_standard_response(res: Any) -> bool:
    """Detect if result already matches CVA's standardize_response shape."""
    if not isinstance(res, dict):
        return False
    return set(res.keys()) >= {"status", "data", "error"}
import threading
import inspect

# Use the unified Tool model + policy helpers
from tool_types import Tool, derive_task_type, validate_role_task_assignment
from config_manager import get_config
import tools
import sandbox_toolsmith

# Logging
logger = logging.getLogger("CatalystLogger")
log = logging.getLogger("ToolRegistry")

# ---------------------------------------------------------------------
# Optional integrations (import with graceful fallbacks)
# ---------------------------------------------------------------------
_PrometheusMetrics = None
_PROM_IMPORT_ERR = None
try:
    from integrations.prometheus_tool import PrometheusMetrics as _PrometheusMetrics  # type: ignore
except Exception as e:
    _PROM_IMPORT_ERR = e

_K8sActions = None
_K8S_IMPORT_ERR = None
try:
    from integrations.k8s_actions_tool import K8sActions as _K8sActions  # type: ignore
except Exception as e:
    _K8S_IMPORT_ERR = e

_OpsPolicyEngine = None
_POLICY_IMPORT_ERR = None
try:
    from core.ops_policy_engine import OpsPolicyEngine as _OpsPolicyEngine  # type: ignore
except Exception as e:
    _POLICY_IMPORT_ERR = e

# ---------------------------------------------------------------------
# Fallback/null implementations so imports never explode
# ---------------------------------------------------------------------
class _NullProm:
    def __init__(self, *a, **kw):
        if _PROM_IMPORT_ERR:
            log.warning("Prometheus not configured (%s) — using Null metrics client.", _PROM_IMPORT_ERR)

    def cpu_percent_avg_5m(self, instance: Optional[str] = None) -> float:
        return 0.0

    def http_p95_ms_5m(self, service_label: str) -> float:
        return 0.0

class _NullK8s:
    def __init__(self, *a, dry_run: bool = True, **kw):
        self.dry_run = dry_run
        if _K8S_IMPORT_ERR:
            log.warning("K8s client not available (%s) — using Null actions client.", _K8S_IMPORT_ERR)

    def restart_deployment(self, namespace: str, name: str) -> Dict[str, Any]:
        return {
            "action": "k8s_restart",
            "namespace": namespace,
            "deployment": name,
            "dry_run": True,
            "status": "error",
            "error": f"k8s not configured: {_K8S_IMPORT_ERR}",
        }

    def scale_deployment(self, namespace: str, name: str, replicas: int) -> Dict[str, Any]:
        return {
            "action": "k8s_scale",
            "namespace": namespace,
            "deployment": name,
            "replicas": replicas,
            "dry_run": True,
            "status": "error",
            "error": f"k8s not configured: {_K8S_IMPORT_ERR}",
        }

class _MiniOpsPolicyEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger("MiniOpsPolicyEngine")
        self._last_action_times: Dict[str, float] = {}
        self._action_counts: Dict[str, List[float]] = {}

    def eval_threshold_rule(self, rule: Dict[str, Any], metric_value: float, now: Optional[float] = None) -> Dict[str, Any]:
        now = now or time.time()
        rid = rule.get("id", "rule")
        op = rule.get("op", ">")
        thr = float(rule.get("threshold", 0))
        approval = rule.get("approval", "human")
        cooldown = int(rule.get("cooldown_seconds", 0))
        budget = int(rule.get("change_budget_per_hour", 999))

        def _ok(v: float) -> bool:
            return (v > thr) if op == ">" else (v < thr)

        last_ts = self._last_action_times.get(rid, 0.0)
        if cooldown and (now - last_ts < cooldown):
            return {"allow": False, "reason": f"cooldown {cooldown}s active", "needs_approval": False}

        window = 3600
        ts_list = [t for t in self._action_counts.get(rid, []) if now - t < window]
        if len(ts_list) >= budget:
            return {"allow": False, "reason": f"budget exhausted ({budget}/h)", "needs_approval": False}

        if not _ok(metric_value):
            return {"allow": False, "reason": "threshold not met", "needs_approval": False}

        return {"allow": True, "needs_approval": (approval != "auto"), "reason": "threshold triggered"}

    def record_action(self, rule_id: str, when: Optional[float] = None):
        when = when or time.time()
        self._last_action_times[rule_id] = when
        self._action_counts.setdefault(rule_id, []).append(when)

# Resolve which classes to use
PrometheusCls = _PrometheusMetrics or _NullProm
K8sActionsCls = _K8sActions or _NullK8s
OpsPolicyEngineCls = _OpsPolicyEngine or _MiniOpsPolicyEngine

# ---------------------------------------------------------------------
# Singletons (robust init with graceful fallbacks)
# ---------------------------------------------------------------------
try:
    PROM = PrometheusCls(logger=log)
except Exception as e:
    log.warning("Prometheus init failed (%s) — using Null metrics client.", e)
    PROM = _NullProm(logger=log)

try:
    K8S = K8sActionsCls(logger=log, dry_run=False)
except Exception as e:
    log.warning("K8s init failed (%s) — using Null actions client.", e)
    K8S = _NullK8s(logger=log, dry_run=True)

try:
    POL = OpsPolicyEngineCls(logger=log)
except Exception as e:
    log.warning("Policy engine init failed (%s) — using minimal in-process engine.", e)
    POL = _MiniOpsPolicyEngine(logger=log)

APPROVAL_MODE = os.getenv("CVA_APPROVAL_MODE", "human").strip().lower()

# ---------------------------------------------------------------------
# Import concrete tool functions via module (so names are qualified)
# ---------------------------------------------------------------------
import tools  # production tool funcs

def _tf(name: str):
    """Resolve a tool function from tools module; raise clear error if missing."""
    fn = getattr(tools, name, None)
    if fn is None:
        raise RuntimeError(f"Missing tool function: tools.{name}")
    return fn

# -----------------------------
# Helpers
# -----------------------------
_PLACEHOLDER_STRINGS: Set[str] = {
    "", " ", "string", "placeholder", "tbd", "todo", "none", "null", "n/a", "na"
}
_REDACT_KEYS = {"api_key", "apikey", "auth", "authorization", "password", "token", "secret"}

def _is_placeholder(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in _PLACEHOLDER_STRINGS
    return False

def _looks_like_url(u: str) -> bool:
    try:
        p = urlparse((u or "").strip())
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False

def _redact(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        out[k] = "***" if k.lower() in _REDACT_KEYS else v
    return out

def _coerce_types_per_schema(args: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return args
    props = (schema or {}).get("properties", {})
    coerced = dict(args)
    for k, spec in props.items():
        if k not in coerced:
            continue
        v = coerced[k]
        t = spec.get("type")
        try:
            if t == "string" and isinstance(v, str):
                coerced[k] = v.strip()
            elif t == "number":
                coerced[k] = float(v) if not isinstance(v, (int, float)) and str(v).strip() else float(v)
            elif t == "integer":
                coerced[k] = int(float(v)) if not isinstance(v, int) else v
            elif t == "boolean":
                if isinstance(v, str):
                    coerced[k] = v.strip().lower() in {"true", "1", "yes", "y"}
        except Exception:
            pass
    return coerced

def _merge_defaults(args: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(args or {})
    props = (schema or {}).get("properties", {})
    for k, spec in props.items():
        if k not in out and "default" in spec:
            out[k] = spec["default"]
    return out

def _light_jsonschema_validate(args: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
    if not schema:
        return True, ""
    props = schema.get("properties", {})
    required = schema.get("required", [])

    for r in required:
        if r not in args or args[r] in (None, "", [], {}):
            return False, f"Missing required arg '{r}'."

    for k, v in args.items():
        spec = props.get(k)
        if not spec:
            continue
        t = spec.get("type")
        if t:
            if t == "string" and not isinstance(v, str):
                return False, f"Arg '{k}' must be string."
            if t == "number" and not isinstance(v, (int, float)):
                return False, f"Arg '{k}' must be number."
            if t == "integer" and not isinstance(v, int):
                if not (isinstance(v, float) and float(v).is_integer()):
                    return False, f"Arg '{k}' must be integer."
            if t == "array" and not isinstance(v, list):
                return False, f"Arg '{k}' must be array."
            if t == "object" and not isinstance(v, dict):
                return False, f"Arg '{k}' must be object."
        if "enum" in spec and v not in spec["enum"]:
            return False, f"Arg '{k}' must be one of {spec['enum']!r}."
        if "minimum" in spec and isinstance(v, (int, float)) and v < spec["minimum"]:
            return False, f"Arg '{k}' must be >= {spec['minimum']}."
        if "maximum" in spec and isinstance(v, (int, float)) and v > spec["maximum"]:
            return False, f"Arg '{k}' must be <= {spec['maximum']}."
    return True, ""

# -----------------------------
# JSON Schemas (unchanged)
# -----------------------------
GET_SYSTEM_CPU_LOAD_PARAMS = {
    "type": "object",
    "properties": {
        "time_interval_seconds": {
            "type": "number",
            "description": "Seconds to wait per measurement (float OK). 0 = non-blocking delta.",
            "default": 0.5,
            "minimum": 0
        },
        "samples": {
            "type": "integer",
            "description": "Number of measurements to average for smoothing.",
            "default": 3,
            "minimum": 1
        },
        "per_core": {
            "type": "boolean",
            "description": "Return per-core percentages instead of system-wide average.",
            "default": False
        }
    },
    "required": [],
    "additionalProperties": False
}

GET_SYSTEM_RESOURCE_USAGE_PARAMS = {"type": "object", "properties": {}, "required": []}

DISK_USAGE_PARAMS = {
    "type": "object",
    "properties": {"path": {"type": "string", "description": "Filesystem path to check.", "default": "/"}},
    "required": []
}

TOP_PROCESSES_PARAMS = {
    "type": "object",
    "properties": {"limit": {"type": "integer", "minimum": 1, "default": 10}},
    "required": []
}

INITIATE_NETWORK_SCAN_PARAMS = {
    "type": "object",
    "properties": {
        "target_ip": {"type": "string", "description": "The IP address or hostname to scan."},
        "scan_type": {"type": "string", "description": "Scan kind.", "enum": ["full_port_scan", "ping_sweep", "vulnerability_scan"], "default": "ping_sweep"},
    },
    "required": ["target_ip"],
}

_PROM_QUERY_PARAMS = {
    "type": "object",
    "properties": {
        "query":     {"type": "string", "description": "PromQL instant query (e.g., 'up')"},
        "timeout_s": {"type": "integer", "minimum": 1, "maximum": 60, "default": 10}
    },
    "required": ["query"],
    "additionalProperties": False,
}

_PROM_RANGE_QUERY_PARAMS = {
    "type": "object",
    "properties": {
        "query":     {"type": "string", "description": "PromQL range query"},
        "start_s":   {"type": "integer", "description": "Unix seconds (start)"},
        "end_s":     {"type": "integer", "description": "Unix seconds (end; must be > start)"},
        "step_s":    {"type": "integer", "minimum": 1, "default": 15, "description": "Step in seconds"},
        "timeout_s": {"type": "integer", "minimum": 1, "maximum": 60, "default": 10}
    },
    "required": ["query", "start_s", "end_s"],
    "additionalProperties": False,
}

DEPLOY_RECOVERY_PROTOCOL_PARAMS = {
    "type": "object",
    "properties": {
        "protocol_name": {"type": "string", "description": "Name of the recovery protocol."},
        "target_system_id": {"type": "string", "description": "System ID."},
        "urgency_level": {"type": "string", "description": "Urgency level.", "enum": ["low", "medium", "high", "critical"], "default": "medium"},
    },
    "required": ["protocol_name", "target_system_id"],
}

UPDATE_RESOURCE_ALLOCATION_PARAMS = {
    "type": "object",
    "properties": {
        "resource_type": {"type": "string", "description": "e.g., cpu, memory"},
        "target_agent_name": {"type": "string", "description": "Agent to modify."},
        "new_allocation_percentage": {"type": "number", "description": "0-100 or 0.0–1.0 (auto-scaled)", "minimum": 0.0},
    },
    "required": ["resource_type", "target_agent_name", "new_allocation_percentage"],
}

GET_ENVIRONMENTAL_DATA_PARAMS = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "Logical location/sensor group."},
        "data_type": {"type": "string", "description": "Which metric to return or 'all'.", "enum": ["all","temperature_celsius","humidity_percent","air_quality_index"], "default": "all"},
    },
    "required": [],
}

ANALYZE_THREAT_SIGNATURE_PARAMS = {
    "type": "object",
    "properties": {
        "signature": {"type": "string", "description": "Threat signature (e.g., CVE-2023-1234)."},
        "source_ip": {"type": "string", "description": "Associated source IP"}
    },
    "required": ["signature"],
}

ISOLATE_NETWORK_SEGMENT_PARAMS = {
    "type": "object",
    "properties": {
        "segment_id": {"type": "string"},
        "reason": {"type": "string"}
    },
    "required": ["segment_id", "reason"]
}

WEB_SEARCH_PARAMS = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
READ_WEBPAGE_PARAMS = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}

UPDATE_WORLD_MODEL_PARAMS = {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}
QUERY_LTM_PARAMS = {"type": "object", "properties": {"query_text": {"type": "string"}}, "required": ["query_text"]}
ANALYZE_TEXT_SENTIMENT_PARAMS = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
CREATE_PDF_PARAMS = {"type": "object", "properties": {"filename": {"type": "string"}, "text_content": {"type": "string"}}, "required": ["filename", "text_content"]}
SHUFFLE_ROLES_PARAMS = {"type": "object", "properties": {"stagnant_agents": {"type": "array", "items": {"type": "string"}}}, "required": ["stagnant_agents"]}

GET_CPU_AVG_5M_PROM_PARAMS = {"type": "object", "properties": {"instance": {"type": "string"}}, "required": []}
GET_HTTP_P95_MS_PROM_PARAMS = {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}

K8S_RESTART_PARAMS = {
    "type": "object",
    "properties": {
        "namespace": {"type": "string", "description": "Kubernetes namespace."},
        "name":      {"type": "string", "description": "Deployment name (alias: 'deployment')."},
        "deployment":{"type": "string", "description": "Alias for 'name'."},
        "approval":  {"type": "string", "enum": ["human", "auto"], "default": "human"},
    },
    "required": ["namespace"]
}

K8S_SCALE_PARAMS = {
    "type": "object",
    "properties": {
        "namespace": {"type": "string", "description": "Kubernetes namespace."},
        "name":      {"type": "string", "description": "Deployment name (alias: 'deployment')."},
        "deployment":{"type": "string", "description": "Alias for 'name'."},
        "replicas":  {"type": "integer", "description": "Target replicas.", "minimum": 1},
        "approval":  {"type": "string", "enum": ["human", "auto"], "default": "human"},
    },
    "required": ["namespace", "replicas"]
}

K8S_POD_METRICS_PARAMS = {
    "type": "object",
    "properties": {
        "namespace": {"type": "string", "description": "Kubernetes namespace (omit for all)."},
        "selector":  {"type": "string", "description": "Label selector, e.g. app=web"},
        "limit":     {"type": "integer", "minimum": 1, "default": 50}
    },
    "required": [],
    "additionalProperties": False
}

POLICY_EVAL_PARAMS = {"type": "object", "properties": {"rule": {"type": "object"}, "metric_value": {"type": "number"}}, "required": ["rule", "metric_value"]}
REDACT_PII_PARAMS = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
GENERATE_REPORT_PDF_PARAMS = {"type": "object", "properties": {"title": {"type": "string"}, "sections": {"type": "array", "items": {"type": "object"}}}, "required": ["title", "sections"]}

HASH_TEXT_PARAMS = {"type": "object", "properties": {"text": {"type": "string"}, "algorithm": {"type": "string", "enum": ["md5","sha1","sha256","sha512"], "default": "sha256"}}, "required": ["text"]}
EXTRACT_IOCS_PARAMS = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

# Tool-specific normalizers
def _normalize_pdf_args(a: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    if "text_content" not in out and "content" in out:
        out["text_content"] = out.pop("content")
    fn = out.get("filename")
    if isinstance(fn, str):
        fn = fn.strip()
        if fn.lower().endswith(".pdf"):
            fn = fn[:-4]
        out["filename"] = fn
    return out

def _normalize_k8s_deploy_args(a: dict) -> dict:
    a = dict(a or {})
    if not a.get("deployment") and a.get("name"):
        a["deployment"] = a["name"]
    if "replicas" in a:
        try:
            a["replicas"] = int(a["replicas"])
        except Exception:
            pass
    return a

# -----------------------------
# Tool Registry
# -----------------------------
class ToolRegistry:
    def __init__(self, db: Any = None):
        self._tools: Dict[str, Tool] = {}
        self._aliases: Dict[str, str] = {}
        self._cooldown_lock = threading.Lock()
        self._failure_lock = threading.Lock()
        # Track consecutive failures and circuit breaker windows
        self._failure_counts: Dict[str, int] = {}
        self._last_failure_ts: Dict[str, float] = {}
        self._broken_until: Dict[str, float] = {}
        self.db = db
        cfg = get_config()
        timeouts_cfg = cfg.get("tool_timeouts", {}) if isinstance(cfg, dict) else {}
        self._per_tool_timeouts = (timeouts_cfg.get("per_tool") or {})
        self.toolsmith_enabled = bool((cfg.get("features", {}) if isinstance(cfg, dict) else {}).get("toolsmith_enabled", False))
        self._initialize_default_tools()

    # --- Registration ---
    def _initialize_default_tools(self) -> None:
        _K8S_POD_METRICS_PARAMS = globals().get("K8S_POD_METRICS_PARAMS") or {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Kubernetes namespace (omit for all)"},
                "selector":  {"type": "string", "description": "Label selector, e.g. app=web"},
                "limit":     {"type": "integer", "minimum": 1, "default": 50}
            },
            "required": [],
            "additionalProperties": False,
        }

        tools_to_register: List[Tool] = [
            # Kubernetes / Observability
            Tool(
                "kubernetes_pod_metrics",
                "Get CPU and memory usage for Kubernetes pods via `kubectl top`.",
                _K8S_POD_METRICS_PARAMS,
                _tf("kubernetes_pod_metrics_tool"),
                cooldown_seconds=2.0,
                task_type="Observation",
                roles_allowed={"Observer"},
            ),

            # Kubernetes Scale (Actuation)
            Tool(
                "k8s_scale",
                "Kubernetes: scale a Deployment to N replicas.",
                K8S_SCALE_PARAMS,
                lambda **kw: (
                    {"status": "awaiting_approval",
                     "action": "k8s_scale",
                     "namespace": kw["namespace"],
                     "deployment": (kw.get("deployment") or kw.get("name")),
                     "replicas": int(kw["replicas"])}
                    if (kw.get("approval", "human").lower() != "auto" and APPROVAL_MODE != "auto")
                    else K8S.scale_deployment(
                        namespace=kw["namespace"],
                        name=(kw.get("deployment") or kw.get("name")),
                        replicas=int(kw["replicas"])
                    )
                ),
                normalizer=_normalize_k8s_deploy_args,
                validator=lambda a: (
                    not _is_placeholder(a.get("namespace"))
                    and (bool(a.get("name")) or bool(a.get("deployment")))
                    and isinstance(a.get("replicas"), int)
                    and a["replicas"] >= 1
                ),
                cooldown_seconds=10.0,
                task_type="Actuation",
                roles_allowed={"Worker"},
            ),

            # Kubernetes restart
            Tool(
                "k8s_restart",
                "Kubernetes: rollout restart a Deployment (adds annotation to pod template).",
                K8S_RESTART_PARAMS,
                lambda **kw: (
                    {"status": "awaiting_approval",
                     "action": "k8s_restart",
                     "namespace": kw["namespace"],
                     "deployment": (kw.get("deployment") or kw.get("name"))}
                    if (kw.get("approval", "human").lower() != "auto" and APPROVAL_MODE != "auto")
                    else K8S.restart_deployment(namespace=kw["namespace"], name=(kw.get("deployment") or kw.get("name")))
                ),
                task_type="Actuation",
                roles_allowed={"Worker"},
                normalizer=_normalize_k8s_deploy_args,
                validator=lambda a: (not _is_placeholder(a.get("namespace"))
                                     and (bool(a.get("name")) or bool(a.get("deployment")))),
                cooldown_seconds=10.0,
            ),

            Tool(
                "check_network_connectivity",
                "Check if a pod can reach a service. Detects network policy blocks and connectivity issues.",
                {
                    "type": "object",
                    "properties": {
                        "source_pod": {"type": "string", "description": "Pod name to test from"},
                        "target_service": {"type": "string", "description": "Service name or IP to reach"},
                        "namespace": {"type": "string", "default": "default", "description": "Namespace"},
                        "timeout": {"type": "integer", "default": 5, "description": "Timeout in seconds"}
                    },
                    "required": ["source_pod", "target_service"],
                    "additionalProperties": False,
                },
                _tf("check_network_connectivity"),
                task_type="Observation",
                roles_allowed={"Observer", "Security", "Planner"},
            ),

            Tool(
                "watch_k8s_audit_events",
                "Monitor for RBAC violations, unauthorized access attempts, and secret usage. Detects forbidden operations and sensitive data access.",
                {
                    "type": "object",
                    "properties": {
                        "minutes": {"type": "integer", "default": 5, "description": "Minutes back to check"},
                        "event_types": {"type": "array", "items": {"type": "string"}, "description": "Filter event types"}
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                _tf("watch_k8s_audit_events"),
                task_type="Observation",
                roles_allowed={"Observer", "Security", "Planner"},
            ),

            Tool(
                "watch_k8s_events",
                "Monitor Kubernetes cluster events. Detects pod kills, crashes, OOMs, restarts, scaling events. Use for incident detection.",
                {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string", "default": "all", "description": "Namespace to watch, or 'all' for all namespaces"},
                        "minutes": {"type": "integer", "default": 5, "description": "How many minutes back to look for events"}
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                _tf("watch_k8s_events"),
                task_type="Observation",
                roles_allowed={"Observer", "Security", "Planner"},
            ),

            Tool(
                "get_pod_status",
                "Get current status of all pods. Detects CrashLoopBackOff, Pending, Failed, OOMKilled pods. Use for health checks.",
                {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string", "default": "default", "description": "Namespace to check, or 'all' for all namespaces"}
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                _tf("get_pod_status"),
                task_type="Observation",
                roles_allowed={"Observer", "Security", "Planner"},
            ),

            # Responsiveness
            Tool("measure_responsiveness", "Measure system responsiveness by timing a harmless command.", {}, _tf("measure_responsiveness_tool"),
                 cooldown_seconds=0.0, task_type="Observation", roles_allowed={"Observer", "Security"}),

            # System & OS
            Tool("get_system_cpu_load", "Measure real CPU utilization.", GET_SYSTEM_CPU_LOAD_PARAMS, _tf("get_system_cpu_load_tool"),
                 task_type="Observation", roles_allowed={"Observer", "Worker"}),
            Tool("get_system_resource_usage", "Checks system-wide CPU and memory load percentages.", GET_SYSTEM_RESOURCE_USAGE_PARAMS, _tf("get_system_resource_usage_tool"),
                 task_type="Observation", roles_allowed={"Observer", "Worker"}),
            Tool("top_processes", "Top processes by CPU% (two-pass accurate sampling).", TOP_PROCESSES_PARAMS, _tf("top_processes_tool"),
                 task_type="Observation", roles_allowed={"Observer", "Security"}),
            Tool("disk_usage", "Filesystem usage for a path.", DISK_USAGE_PARAMS, _tf("disk_usage_tool"),
                 task_type="GenericTask", roles_allowed={"Observer","Worker","Security"}),

            # Security / Networking
            Tool("initiate_network_scan", "Initiates a network scan on a specified IP address.", INITIATE_NETWORK_SCAN_PARAMS, _tf("initiate_network_scan_tool"),
                 validator=lambda a: not _is_placeholder(a.get("target_ip")), task_type="GenericTask", roles_allowed={"Security","Observer"}),
            Tool("deploy_recovery_protocol", "Deploys a pre-defined recovery protocol to a target system.", DEPLOY_RECOVERY_PROTOCOL_PARAMS, _tf("deploy_recovery_protocol_tool"),
                 validator=lambda a: (not _is_placeholder(a.get("protocol_name")) and not _is_placeholder(a.get("target_system_id"))),
                 task_type="GenericTask", roles_allowed={"Security","Worker"}),
            Tool("update_resource_allocation", "Adjusts the resource allocation for a given agent (idempotent).", UPDATE_RESOURCE_ALLOCATION_PARAMS, _tf("update_resource_allocation_tool"),
                 validator=lambda a: (not _is_placeholder(a.get("resource_type")) and not _is_placeholder(a.get("target_agent_name")) and (a.get("new_allocation_percentage") is not None)),
                 task_type="ResourceTuning", roles_allowed={"Worker"}),
            Tool("analyze_threat_signature", "Analyzes a known threat signature to determine its risk level.", ANALYZE_THREAT_SIGNATURE_PARAMS, _tf("analyze_threat_signature_tool"),
                 validator=lambda a: not _is_placeholder(a.get("signature")), task_type="GenericTask", roles_allowed={"Security"}),
            Tool("isolate_network_segment", "Isolates a network segment to prevent a threat from spreading.", ISOLATE_NETWORK_SEGMENT_PARAMS, _tf("isolate_network_segment_tool"),
                 validator=lambda a: (not _is_placeholder(a.get("segment_id")) and not _is_placeholder(a.get("reason"))),
                 cooldown_seconds=10.0, task_type="GenericTask", roles_allowed={"Security"}),

            # Info Retrieval
            Tool("web_search", "Performs a web search using SerpApi.", WEB_SEARCH_PARAMS, _tf("web_search_tool"),
                 task_type="InformationRetrieval", roles_allowed={"Planner", "Observer", "Security"}, redact_fields={"api_key"},
                 validator=lambda a: not _is_placeholder(a.get("query"))),
            Tool("read_webpage", "Reads the textual content of a webpage from a URL.", READ_WEBPAGE_PARAMS, _tf("read_webpage_tool"),
                 task_type="InformationRetrieval", roles_allowed={"Planner", "Observer"},
                 validator=lambda a: isinstance(a.get("url"), str) and _looks_like_url(a.get("url"))),

            Tool("redact_pii", "Redact PII from text.", REDACT_PII_PARAMS, _tf("redact_pii_tool"),
                 validator=lambda a: not _is_placeholder(a.get("text")), task_type="GenericTask", roles_allowed={"Planner","Observer"}),

            # World/Memory
            Tool("update_world_model", "Updates the swarm's shared world-state.", UPDATE_WORLD_MODEL_PARAMS, _tf("update_world_model_tool"),
                 validator=lambda a: (not _is_placeholder(a.get("key")) and not _is_placeholder(a.get("value"))),
                 task_type="GenericTask", roles_allowed={"Planner","Observer","Worker"}),
            Tool("query_long_term_memory", "Searches the agent's long-term memory.", QUERY_LTM_PARAMS, _tf("query_long_term_memory_tool"),
                 validator=lambda a: not _is_placeholder(a.get("query_text")), task_type="GenericTask", roles_allowed={"Planner","Observer"}),

            # Text / Reports
            Tool("analyze_text_sentiment", "Analyzes text sentiment (positive/negative).", ANALYZE_TEXT_SENTIMENT_PARAMS, _tf("analyze_text_sentiment_tool"),
                 validator=lambda a: not _is_placeholder(a.get("text")), task_type="GenericTask", roles_allowed={"Planner","Observer"}),
            Tool("create_pdf", "Creates a PDF from text content and saves it.", CREATE_PDF_PARAMS, _tf("create_pdf_tool"),
                 normalizer=_normalize_pdf_args,
                 validator=lambda a: (not _is_placeholder(a.get("filename")) and not _is_placeholder(a.get("text_content"))),
                 task_type="GenericTask", roles_allowed={"Planner","Worker"}),
            Tool("shuffle_roles_and_tasks", "Generates exploratory tasks for stagnant agents.", SHUFFLE_ROLES_PARAMS, _tf("shuffle_roles_and_tasks_tool"),
                 validator=lambda a: isinstance(a.get("stagnant_agents"), list) and len(a.get("stagnant_agents")) > 0,
                 task_type="GenericTask", roles_allowed={"Planner"}),

            # Testing / diagnostics
            Tool("long_sleep", "Sleep for N seconds (testing timeout handling).",
                 {"type": "object", "properties": {"seconds": {"type": "integer", "minimum": 1, "maximum": 900, "default": 60}}},
                 _tf("long_sleep_tool"),
                 task_type="GenericTask",
                 roles_allowed={"Worker", "Planner", "Observer"},
                 timeout_seconds=0),

            Tool("system_diagnostics", "Return system diagnostics (CPU/mem/threads/agents/logs/db).",
                 {},
                 _tf("system_diagnostics_tool"),
                 task_type="Observation",
                 roles_allowed={"Planner", "Observer"},
                 timeout_seconds=10),

            Tool("self_test", "Run a quick CVA self-test (DB, registry, trivial tool, agents).",
                 {},
                 _tf("self_test_tool"),
                 task_type="Observation",
                 roles_allowed={"Planner", "Observer"},
                 timeout_seconds=10),

            Tool("restart_agent", "Restart a named agent via AgentFactory.",
                 {"type": "object", "properties": {"agent_name": {"type": "string"}}},
                 _tf("restart_agent_tool"),
                 task_type="GenericTask",
                 roles_allowed={"Planner", "Worker"}),

            Tool("tool_breaker_status", "Inspect tool circuit breaker state (failure counts and backoffs).",
                 {},
                 lambda **kw: self._breaker_status_tool(),
                 task_type="Observation",
                 roles_allowed={"Planner", "Observer", "Worker"}),

            Tool("toolsmith_generate", "Generate a sandboxed tool on the fly for unmet tasks.",
                 {"type": "object", "properties": {"task": {"type": "string"}, "code_hint": {"type": "string"}}},
                 _tf("toolsmith_generate"),
                 task_type="GenericTask",
                 roles_allowed={"Planner", "Worker"}),

            # Environment (synthetic)
            Tool("get_environmental_data", "Fetches synthetic environmental sensor data.", GET_ENVIRONMENTAL_DATA_PARAMS, _tf("get_environmental_data_tool"),
                 task_type="GenericTask", roles_allowed={"Observer"}),

            # Prometheus (read-only)
            Tool("get_cpu_avg_5m", "Prometheus: average CPU percent over ~5m window (100 - idle).", GET_CPU_AVG_5M_PROM_PARAMS,
                 lambda **kw: {"cpu_percent_avg_5m": PROM.cpu_percent_avg_5m(instance=kw.get("instance"))},
                 cooldown_seconds=2.0, task_type="Observation", roles_allowed={"Observer"}),
            Tool("get_http_p95_ms", "Prometheus: HTTP p95 latency (ms) over 5m for a service label.", GET_HTTP_P95_MS_PROM_PARAMS,
                 lambda **kw: {"http_p95_ms_5m": PROM.http_p95_ms_5m(service_label=kw["service"])},
                 validator=lambda a: not _is_placeholder(a.get("service")),
                 cooldown_seconds=2.0, task_type="Observation", roles_allowed={"Observer"}),

            Tool("prometheus_query", "Prometheus instant query (PromQL). Returns raw API JSON + small summary.", _PROM_QUERY_PARAMS, _tf("prometheus_query_tool"),
                 validator=lambda a: (isinstance(a.get("query"), str) and len(a["query"].strip()) > 0),
                 cooldown_seconds=1.0, task_type="Observation", roles_allowed={"Observer", "Planner", "Worker"}),

            Tool("prometheus_range_query", "Prometheus range query over [start_s, end_s] with step.", _PROM_RANGE_QUERY_PARAMS, _tf("prometheus_range_query_tool"),
                 validator=lambda a: (isinstance(a.get("query"), str) and len(a["query"].strip()) > 0
                                      and isinstance(a.get("start_s"), int) and isinstance(a.get("end_s"), int)
                                      and a["end_s"] > a["start_s"] and (a.get("step_s", 15) >= 1)),
                 cooldown_seconds=2.0, task_type="Observation", roles_allowed={"Observer", "Planner"}),

            Tool("send_desktop_notification", "Send a desktop notification to the user.", 
                {"title": {"type": "string"}, "message": {"type": "string"}}, 
                _tf("send_desktop_notification_tool"),
                task_type="Notification", 
                roles_allowed={"Notifier", "Planner", "Security"}),
            
            Tool("check_calendar", "Check Google Calendar for events in a time range.", 
                 {"time_min_utc": {"type": "string"}, "time_max_utc": {"type": "string"}},
                 _tf("check_calendar_tool"),
                 task_type="InformationRetrieval",
                 roles_allowed={"Planner", "Observer"}),

            Tool("find_wasteful_deployments", "Find Kubernetes deployments with low resource utilization.",
                 {"namespace": {"type": "string", "default": "default"}},
                 _tf("find_wasteful_deployments_tool"),
                 task_type="Observation",
                 roles_allowed={"Observer", "Worker"}),

            Tool("get_tool_usage_stats", "Get statistics on tool usage.",
                 {},
                 _tf("get_tool_usage_stats_tool"),
                 task_type="Observation",
                 roles_allowed={"Observer", "Planner"}),

            Tool("list_available_tools", "List all available tools in the registry.",
                 {},
                 _tf("list_available_tools_tool"),
                 task_type="Observation",
                 roles_allowed={"Planner", "Observer"}),

            Tool("tool_health_check", "Check health status of all tools.",
                 {},
                 _tf("tool_health_check_tool"),
                 task_type="Observation",
                 roles_allowed={"Observer", "Security"}),

            # Policy eval
            Tool("policy_eval", "Evaluate a threshold rule (approval, cooldown, budgets) against a metric value.", POLICY_EVAL_PARAMS,
                 lambda **kw: POL.eval_threshold_rule(rule=kw["rule"], metric_value=kw["metric_value"]),
                 validator=lambda a: isinstance(a.get("rule"), dict) and isinstance(a.get("metric_value"), (int, float)),
                 task_type="GenericTask", roles_allowed={"Planner","Observer"}),

            # Utility / Security text tools
            Tool("hash_text", "Hash text using a chosen algorithm.", HASH_TEXT_PARAMS, _tf("hash_text_tool"),
                 task_type="GenericTask", roles_allowed={"Observer","Security","Worker","Planner"}),
            Tool("extract_iocs", "Extract IoCs (IPs/domains/hashes) from text.", EXTRACT_IOCS_PARAMS, _tf("extract_iocs_tool"),
                 task_type="GenericTask", roles_allowed={"Security","Observer"}),
        ]

        for tool in tools_to_register:
            self.register_tool(tool)

    def register_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_alias(self, alias: str, canonical_name: str) -> None:
        if canonical_name not in self._tools:
            raise KeyError(f"Cannot alias to unknown tool '{canonical_name}'.")
        self._aliases[alias] = canonical_name

    # --- Lookup ---
    def _resolve_name(self, name: str) -> str:
        return self._aliases.get(name, name)

    def has_tool(self, tool_name: str) -> bool:
        return self._resolve_name(tool_name) in self._tools

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        return self._tools.get(self._resolve_name(tool_name))

    def list_tool_names(self) -> List[str]:
        return sorted(self._tools.keys())

    def get_available_tools(self) -> Set[str]:
        return set(self._tools.keys())

    # --- LLM prompt helpers ---
    def get_all_tool_specs(self) -> List[dict]:
        out = []
        for t in self._tools.values():
            out.append({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "cooldown_seconds": getattr(t, "cooldown_seconds", 0.0),
                "task_type": getattr(t, "task_type", "GenericTask"),
                "roles_allowed": list(getattr(t, "roles_allowed", [])) if hasattr(t, "roles_allowed") else [],
            })
        return out

    def get_tool_instructions(self) -> str:
        lines = []
        for tool in self._tools.values():
            schema = tool.parameters or {}
            props = (schema.get("properties") or {})
            required = set(schema.get("required") or [])
            arg_details = {
                name: {"type": spec.get("type", "any"), **({"default": spec["default"]} if "default" in spec else {})}
                for name, spec in props.items()
            }
            lines.append(
                f"- Tool Name: {tool.name}\n"
                f"  Description: {tool.description}\n"
                f"  TaskType: {tool.task_type}\n"
                f"  RolesAllowed: {sorted(tool.roles_allowed) if tool.roles_allowed else 'any'}\n"
                f"  Args(required): {sorted(required)}\n"
                f"  Args(optional): {sorted(set(props.keys()) - required)}\n"
                f"  ArgTypes/Defaults: {arg_details}"
            )
        return "\n".join(lines)

    # --- Validation + execution ---
    def validate_args(self, tool_name: str, args: Dict[str, Any]) -> bool:
        tool = self.get_tool(tool_name)
        if not tool:
            return False
        a = _merge_defaults(args or {}, tool.parameters)
        a = _coerce_types_per_schema(a, tool.parameters)
        ok, _ = _light_jsonschema_validate(a, tool.parameters)
        if not ok:
            return False
        for k, v in a.items():
            if isinstance(v, str) and _is_placeholder(v):
                return False
        if "url" in (tool.parameters.get("properties") or {}):
            u = a.get("url")
            if isinstance(u, str) and not _looks_like_url(u):
                return False
        if tool.validator and not tool.validator(a):
            return False
        return True

    def call(self, tool_name: str, **kwargs) -> Any:
        t = self.get_tool(tool_name)
        if not t:
            raise KeyError(f"Unknown tool '{tool_name}'")
        return t.func(**kwargs)

    def safe_call(self, tool_name: str, timeout_seconds: Optional[int] = None, **kwargs) -> Any:
        tool = self.get_tool(tool_name)
        canonical = tool.name
        now = time.time()

        # Circuit breaker: short-circuit if tool is marked broken and still in backoff
        with self._failure_lock:
            broken_until = self._broken_until.get(canonical, 0.0)
            last_fail = self._last_failure_ts.get(canonical, 0.0)
            # Reset stale counters after 5 minutes without failures
            if self._failure_counts.get(canonical, 0) > 0 and (now - last_fail) > 300:
                self._failure_counts[canonical] = 0
            if broken_until and now < broken_until:
                wait_s = int(broken_until - now)
                logger.warning(f"[TOOL BREAKER] '{canonical}' short-circuited; wait ~{wait_s}s before retry.")
                return {
                    "status": "error",
                    "data": None,
                    "error": f"Tool '{canonical}' temporarily disabled after repeated failures. Retry after {wait_s}s.",
                    "summary": None
                }
            elif broken_until and now >= broken_until:
                # Backoff expired; clear breaker and start fresh
                self._broken_until.pop(canonical, None)
                self._failure_counts[canonical] = 0
                logger.info(f"[TOOL BREAKER] '{canonical}' reset after backoff window.")

        # Cooldown check
        if getattr(tool, "cooldown_seconds", 0.0) > 0:
            now = time.time()
            with self._cooldown_lock:
                last = getattr(tool, "_last_called_ts", 0.0)
                if now - last < tool.cooldown_seconds:
                    wait = max(0.0, tool.cooldown_seconds - (now - last))
                    return f"[ERROR] Cooldown active for '{canonical}'. Try again in {wait:.2f}s."
                setattr(tool, "_last_called_ts", now)
        
        # Merge defaults + validate
        args = _merge_defaults(kwargs, tool.parameters)
        args = _coerce_types_per_schema(args, tool.parameters)
        if tool.normalizer:
            args = tool.normalizer(args)
        ok, err = _light_jsonschema_validate(args, tool.parameters)
        if not ok:
            return f"[ERROR] {canonical}: {err}"
        
        # Execute with timeout
        def _execute():
            return tool.func(**args)
        
        # Resolve timeout: caller arg wins; else tool-level setting; 0/negative means no timeout
        eff_timeout = timeout_seconds
        if eff_timeout is None:
            eff_timeout = getattr(tool, "timeout_seconds", 30)
        # Config-driven per-tool override (if provided)
        override = None
        try:
            override = self._per_tool_timeouts.get(canonical)
        except Exception:
            override = None
        if override is not None:
            try:
                eff_timeout = float(override)
            except Exception:
                pass
        if isinstance(eff_timeout, (int, float)) and eff_timeout <= 0:
            eff_timeout = None  # wait indefinitely (agent-level timeouts may still apply)

        t0 = time.time()
        logger.info(f"[TOOL CALL] {canonical} args={_redact(args)} timeout={eff_timeout if eff_timeout is not None else 'none'}")
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_execute)
                result = future.result(timeout=eff_timeout) if eff_timeout is not None else future.result()
            dt = (time.time() - t0) * 1000.0
            logger.info(f"[TOOL OK] {canonical} ({dt:.1f} ms)")
            self._record_tool_success(canonical)
            # Record success metric
            try:
                exec_time = dt / 1000.0
                if hasattr(self, "db") and self.db:
                    self.db.record_metric(
                        metric_type="tool_execution",
                        tool_name=tool_name,
                        value=1.0,  # success
                        metadata={"execution_time": exec_time, "status": "success"}
                    )
            except Exception:
                pass
            
            # Normalize to standard schema if needed
            if _is_standard_response(result):
                return result
            if isinstance(result, dict) and "success" in result and set(result.keys()) <= {"success","data","error"}:
                # legacy success envelope; convert
                status = "ok" if result.get("success") else "error"
                if status != "ok":
                    self._record_tool_failure(canonical)
                return {"status": status, "data": result.get("data"), "error": result.get("error"), "summary": None}
            # Fallback: wrap raw result
            return {"status": "ok", "data": result, "error": None, "summary": None}
            
        except FutureTimeoutError:
            dt = (time.time() - t0) * 1000.0
            logger.error(f"[TOOL TIMEOUT] {canonical} exceeded {eff_timeout}s")
            self._record_tool_failure(canonical)
            return {"status": "error", "data": None, "error": f"timeout after {eff_timeout}s", "summary": None}
            
        except Exception as e:
            dt = (time.time() - t0) * 1000.0
            logger.error(f"[TOOL FAILED] {canonical} in {dt:.1f}ms: {e}")
            self._record_tool_failure(canonical)
            # Record failure metric
            try:
                if hasattr(self, "db") and self.db:
                    self.db.record_metric(
                        metric_type="tool_execution",
                        tool_name=tool_name,
                        value=0.0,  # failure
                        metadata={"error": str(e), "status": "failure"}
                    )
            except Exception:
                pass
            return {"status": "error", "data": None, "error": str(e), "summary": None}

    # --- Circuit breaker helpers ---
    def _record_tool_success(self, tool_name: str) -> None:
        with self._failure_lock:
            if self._broken_until.get(tool_name):
                logger.info(f"[TOOL BREAKER] '{tool_name}' recovered; clearing breaker.")
                try:
                    if getattr(self, "db", None):
                        self.db.record_metric(
                            metric_type="circuit_breaker_reset",
                            value=1.0,
                            tool_name=tool_name,
                            metadata={"reason": "recovered"}
                        )
                except Exception:
                    pass
            self._failure_counts[tool_name] = 0
            self._last_failure_ts.pop(tool_name, None)
            self._broken_until.pop(tool_name, None)

    def _record_tool_failure(self, tool_name: str) -> None:
        now = time.time()
        with self._failure_lock:
            # Reset stale counters after 5 minutes without failures
            last_fail = self._last_failure_ts.get(tool_name, 0.0)
            if last_fail and (now - last_fail) > 300:
                self._failure_counts[tool_name] = 0
            self._last_failure_ts[tool_name] = now
            self._failure_counts[tool_name] = self._failure_counts.get(tool_name, 0) + 1

            if self._failure_counts[tool_name] >= 3:
                # Mark tool as broken for 5 minutes
                self._broken_until[tool_name] = now + 300
                logger.warning(f"[TOOL BREAKER] '{tool_name}' marked broken after {self._failure_counts[tool_name]} consecutive failures; backoff 300s.")
                try:
                    if getattr(self, "db", None):
                        self.db.record_metric(
                            metric_type="circuit_breaker_trip",
                            value=1.0,
                            tool_name=tool_name,
                            metadata={"reason": "max_failures", "cooldown": 300}
                        )
                except Exception:
                    pass

    # --- Diagnostics ---
    def _breaker_status_tool(self) -> Dict[str, Any]:
        """Return breaker state for all tools."""
        now = time.time()
        with self._failure_lock:
            data = []
            for name in self._tools.keys():
                cnt = self._failure_counts.get(name, 0)
                broken_until = self._broken_until.get(name)
                status = "broken" if broken_until and broken_until > now else "healthy"
                if broken_until and broken_until <= now:
                    # expired but not yet cleared
                    status = "expired"
                data.append({
                    "tool": name,
                    "failure_count": cnt,
                    "broken_until": broken_until,
                    "status": status,
                })
        return {
            "status": "ok",
            "data": {"tools": data, "timestamp": now},
            "error": None,
            "summary": "Breaker status snapshot"
        }
# -----------------------------
# Single, importable instance + legacy TOOLS dict
# -----------------------------
from database import cva_db
tool_registry = ToolRegistry(db=cva_db)

def _wrap_safe(name: str):
    return (lambda n: (lambda **kw: tool_registry.safe_call(n, **kw)))(name)

TOOLS = {name: _wrap_safe(name) for name in tool_registry.list_tool_names()}

__all__ = ["tool_registry", "TOOLS", "PROM", "K8S", "POL", "ToolRegistry"]
