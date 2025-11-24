# core/policy.py
from __future__ import annotations
from tool_registry import tool_registry as _registry

"""
Policy layer for planning & execution.

Goals
- Enforce strategic intent (reject aimless/placeholder steps)
- Enforce role → task-type permissions
- Map tool → task-type (auto-imported from tools registry when available)
- Backward compatibility with existing imports (validate_task_intent, validate_role_tool_assignment)
- Provide helpful utilities for diagnostics

This module has zero side effects and no runtime dependencies outside stdlib.
"""

from typing import Optional, Dict, Set, Tuple, Any
import logging
import re

def _policy_resolve_task_type(step: dict) -> str:
    """
    If the step is a ToolRun (or unspecified), use the tool's own task_type
    from the registry; otherwise normalize the provided task_type.
    """
    raw = (step.get("task_type") or "").strip()
    if not raw or raw.lower() == "toolrun":
        tool_name = step.get("tool") or step.get("tool_name") or ""
        t = _registry.get_tool(tool_name) if tool_name else None
        if t and getattr(t, "task_type", None):
            return normalize_task_type(t.task_type)
        return "GenericTask"
    return normalize_task_type(raw)

def _policy_has_tool(tool_name: str) -> bool:
    """Registry-aware tool existence check."""
    return bool(tool_name) and _registry.has_tool(tool_name)

def _policy_canonical_tool(tool_name: str) -> str | None:
    """Return the canonical tool name the registry knows (if any)."""
    t = _registry.get_tool(tool_name) if tool_name else None
    return tool_name if t else None

LOGGER = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Strategic intent resolution (safe + deterministic)
# -------------------------------------------------------------------------

ALLOWED_INTENTS: set[str] = {
    "security_audit",
    "performance_optimization",
    "workflow_optimization",
    "memory_optimization",
    "config_tuning",
    "health_audit",
    "status_reporting",
}

INTENT_SYNONYMS: Dict[str, str] = {
    # security
    "security": "security_audit",
    "threat": "security_audit",
    "vulnerabilities": "security_audit",
    "cve": "security_audit",
    "malware": "security_audit",
    "ioc": "security_audit",
    "statusreporting": "status_reporting",  # tolerate missing underscore / casing

    # performance
    "perf": "performance_optimization",
    "performance": "performance_optimization",
    "cpu": "performance_optimization",
    "latency": "performance_optimization",
    "throughput": "performance_optimization",
    "bottleneck": "performance_optimization",

    # workflow / memory / config / health
    "workflow": "workflow_optimization",
    "memory": "memory_optimization",
    "config": "config_tuning",
    "configuration": "config_tuning",
    "health": "health_audit",
    "environmental": "health_audit",
    "sensor": "health_audit",

    # reporting / status
    "status": "status_reporting",
    "report": "status_reporting",
    "summary": "status_reporting",
    "pdf": "status_reporting",

    # web-ish cues that should land in status/reporting (not too broad)
    "web": "status_reporting",
    "search": "status_reporting",
    "webpage": "status_reporting",
    "read": "status_reporting",
}

# Tool → default intent (if nothing explicit and no mission)
TOOL_DEFAULT_INTENT: dict[str, str] = {
    # observation / perf / health
    "get_system_cpu_load": "performance_optimization",
    "get_system_resource_usage": "performance_optimization",
    "get_environmental_data": "status_reporting",

    # security
    "analyze_threat_signature": "security_audit",
    "initiate_network_scan": "security_audit",
    "isolate_network_segment": "security_audit",
    "deploy_recovery_protocol": "security_audit",

    # information gathering / reporting
    "web_search": "status_reporting",
    "read_webpage": "status_reporting",
    "create_pdf": "status_reporting",
    "update_world_model": "status_reporting",

    # resource mgmt
    "update_resource_allocation": "performance_optimization",
}

# Very small, specific keyword fallbacks (last resort, whole-word match)
KEYWORD_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(threat|ioc|malware|exploit|cve|isolate|quarantine)\b", re.I), "security_audit"),
    (re.compile(r"\b(cpu|latency|throughput|bottleneck|optimi[sz]e|throttle|scale)\b", re.I), "performance_optimization"),
    (re.compile(r"\b(memory|cache|gc)\b", re.I), "memory_optimization"),
    (re.compile(r"\b(config|configuration|tune|parameter)\b", re.I), "config_tuning"),
    (re.compile(r"\b(health|heartbeat|sensor|temperature|environment(al)?|telemetry)\b", re.I), "health_audit"),
    (re.compile(r"\b(report|summary|brief|status|pdf)\b", re.I), "status_reporting"),
]

_PLACEHOLDER_STRINGS = {
    "", " ", "tbd", "todo", "none", "null", "n/a", "na", "unspecified", "no specific intent"
}


def _keyword_intent_from_title(title: str) -> str | None:
    if not title:
        return None
    for pattern, intent in KEYWORD_RULES:
        if pattern.search(title):
            return intent
    return None

def resolve_strategic_intent(
    *,
    step_intent: Optional[str],
    mission_type: Optional[str],
    tool: Optional[str],
    task_type_hint: Optional[str],  # reserved for future heuristics
    title: Optional[str],
) -> Tuple[bool, Optional[str], list[str], Optional[str]]:
    """
    Returns: (intent_ok, intent_value, reasons[], intent_source)
    Precedence:
      1) explicit step_intent (canonical or synonym)
      2) mission_type (canonical or synonym)
      3) tool default (after alias/canonical resolution)
      4) title keyword fallback (tight patterns only)
    """
    reasons: list[str] = []

    # 1) explicit (accept synonyms + ignore placeholders)
    intent = normalize_intent(step_intent)
    if intent in ALLOWED_INTENTS:
        return True, intent, reasons, "step_intent"
    elif step_intent and intent is None:
        # They supplied something, but it wasn't recognized (and not a placeholder)
        low = step_intent.strip().lower()
        if low not in _PLACEHOLDER_STRINGS:
            reasons.append(f"Unknown intent '{step_intent}', ignoring.")

    # 2) parent mission (accept synonyms + ignore placeholders)
    intent = normalize_intent(mission_type)
    if intent in ALLOWED_INTENTS:
        return True, intent, reasons, "mission_type"
    elif mission_type and intent is None:
        low = mission_type.strip().lower()
        if low not in _PLACEHOLDER_STRINGS:
            reasons.append(f"Unknown mission_type '{mission_type}', ignoring.")

    # 3) tool default (resolve alias → canonical tool name first)
    if tool:
        can_tool = _canonical_tool_name(tool)
        td = TOOL_DEFAULT_INTENT.get(can_tool) or TOOL_DEFAULT_INTENT.get(tool)
        if td in ALLOWED_INTENTS:
            return True, td, reasons, "tool_default"
        elif td:
            reasons.append(f"Tool '{tool}' default intent '{td}' not allowed, ignoring.")

    # 4) title fallback (tight keywords)
    intent = _keyword_intent_from_title(title or "")
    if intent:
        return True, intent, reasons, "title_keywords"

    reasons.append("Invalid or missing strategic intent.")
    return False, None, reasons, None

# Back-compat helpers (now thin wrappers over resolve_strategic_intent)
def validate_step_intent(step: dict) -> bool:
    ok, _, _, _ = resolve_strategic_intent(
        step_intent=step.get("intent"),
        mission_type=step.get("mission_type"),
        tool=step.get("tool"),
        task_type_hint=step.get("task_type"),
        title=step.get("title"),
    )
    return ok

def validate_task_intent(title: str) -> bool:
    ok, _, _, _ = resolve_strategic_intent(
        step_intent=None,
        mission_type=None,
        tool=None,
        task_type_hint=None,
        title=title,
    )
    return ok


# -------------------------------------------------------------------------
# Tool → task type mapping
# -------------------------------------------------------------------------

TASK_TYPES: Set[str] = {
    "Observation",
    "SecurityOperation",
    "ResourceMgmt",
    "Reporting",
    "NLP",
    "DataAcquisition",
    "WebAccess",
    "Knowledge",
    "FileOutput",
    "Orchestration",
    "ToolRun",
    "GenericTask",
    "Actuation"
    
}

TOOL_TASK_TYPES: Dict[str, str] = {}
TOOL_ALIASES: Dict[str, str] = {}  # alias -> canonical tool name (optional)


def _bootstrap_tool_mappings() -> None:
    global TOOL_TASK_TYPES, TOOL_ALIASES
    try:
        # Preferred: modern registry (exposed by your upgraded tools.py)
        from tools import TOOL_TASK_TYPES as TT  # type: ignore
        TOOL_TASK_TYPES = dict(TT or {})
        try:
            from tools import TOOL_ALIASES as TA  # type: ignore
            TOOL_ALIASES = dict(TA or {})
        except Exception:
            TOOL_ALIASES = {}
        if not TOOL_TASK_TYPES:
            raise RuntimeError("Empty TOOL_TASK_TYPES")

        # Defensive supplement: make sure these common mappings exist
        try:
            TOOL_TASK_TYPES.update({
                "k8s_scale": "Actuation",
                "k8s_restart": "Actuation",
                "kubernetes_pod_metrics": "Observation",
                "measure_responsiveness": "Observation",
            })
        except Exception:
            pass
        return
    except Exception:
        # Fallback: conservative static map (covers current tool names & short aliases)
        TOOL_ALIASES = {
            # short -> implementation
            "get_system_cpu_load": "get_system_cpu_load_tool",
            "get_system_resource_usage": "get_system_resource_usage_tool",
            "initiate_network_scan": "initiate_network_scan_tool",
            "deploy_recovery_protocol": "deploy_recovery_protocol_tool",
            "update_resource_allocation": "update_resource_allocation_tool",
            "analyze_threat_signature": "analyze_threat_signature_tool",
            "isolate_network_segment": "isolate_network_segment_tool",
            "get_environmental_data": "get_environmental_data_tool",
            "web_search": "web_search_tool",
            "read_webpage": "read_webpage_tool",
            "update_world_model": "update_world_model_tool",
            "query_long_term_memory": "query_long_term_memory_tool",
            "analyze_text_sentiment": "analyze_text_sentiment_tool",
            "create_pdf": "create_pdf_tool",
            "shuffle_roles_and_tasks": "shuffle_roles_and_tasks_tool",
            "k8s_scale": "k8s_scale",
            "k8s_restart": "k8s_restart",
            "kubernetes_pod_metrics": "kubernetes_pod_metrics",
            "measure_responsiveness": "measure_responsiveness"
        }
        TOOL_TASK_TYPES = {
            # implementation name -> task type
            "get_system_cpu_load_tool": "Observation",
            "get_system_resource_usage_tool": "Observation",
            "initiate_network_scan_tool": "SecurityOperation",
            "deploy_recovery_protocol_tool": "SecurityOperation",
            "update_resource_allocation_tool": "ResourceMgmt",
            "analyze_threat_signature_tool": "SecurityOperation",
            "isolate_network_segment_tool": "SecurityOperation",
            "get_environmental_data_tool": "DataAcquisition",
            "web_search_tool": "WebAccess",
            "read_webpage_tool": "WebAccess",
            "update_world_model_tool": "Knowledge",
            "query_long_term_memory_tool": "Knowledge",
            "analyze_text_sentiment_tool": "NLP",
            "create_pdf_tool": "FileOutput",
            "shuffle_roles_and_tasks_tool": "Orchestration",
            # direct names when planner uses them
            "k8s_scale": "Actuation",
            "k8s_restart": "Actuation",
            "kubernetes_pod_metrics": "Observation",
            "measure_responsiveness": "Observation"
        }
def _canonical_tool_name(tool: Optional[str]) -> Optional[str]:
    if not tool or not isinstance(tool, str):
        return None
    t = tool.strip()
    if not t:
        return None
    # Already an impl name?
    if t in TOOL_TASK_TYPES:
        return t
    # Alias?
    if t in TOOL_ALIASES:
        return TOOL_ALIASES[t]
    # If neither, return raw (upstream may still resolve it later)
    return t


def resolve_task_type(tool: Optional[str] = None, step: Optional[dict] = None) -> str:
    """
    Robust resolver for task type.
    - You can pass just the tool name (current utils.py usage)
    - Or pass the whole step (if it includes 'task_type' already)
    """
    # If caller provided a step-embedded task_type, trust it (but keep sane fallback)
    if isinstance(step, dict):
        ttype = step.get("task_type")
        if isinstance(ttype, str) and ttype in TASK_TYPES:
            return ttype
        # Accept common aliases here too
        if isinstance(ttype, str):
            aliased = normalize_task_type(ttype)
            if aliased in TASK_TYPES:
                return aliased

    can_tool = _canonical_tool_name(tool)
    if isinstance(can_tool, str) and can_tool in TOOL_TASK_TYPES:
        return TOOL_TASK_TYPES[can_tool]

    return "GenericTask"

# -------------------------------------------------------------------------
# Roles & permissions (DROP-IN REPLACEMENT)
# -------------------------------------------------------------------------

# Canonical role -> allowed canonical task types
ROLE_ALLOWED_TASKS: Dict[str, Set[str]] = {
    "Observer": {"Observation", "DataAcquisition", "WebAccess", "Knowledge"},
    "Security": {"SecurityOperation", "Observation"},
    "Planner": {"Orchestration", "Reporting"},  # keep tight for now
    "Worker": {
        "GenericTask", "ToolRun", "Observation", "WebAccess", "Knowledge",
        "FileOutput", "NLP", "ResourceMgmt", "DataAcquisition", "Reporting",
        "Actuation",
    },
}

# Task type aliases emitted by other components → canonical types above
TASK_TYPE_ALIASES: Dict[str, str] = {
    # Observability
    "FileOutput": "Reporting",  # treat file creation as reporting
    "SystemObservation": "Observation",
    "EnvironmentObservation": "Observation",

    # Info access
    "InformationRetrieval": "WebAccess",
    "ToolExecution": "ToolRun",

    # Planning / reporting
    "Planning": "Orchestration",
    "StatusReporting": "Reporting",
    "StatusUpdate": "Reporting",
    "Reporting": "Reporting",

    # Resource / config / performance / health
    "ResourceTuning": "ResourceMgmt",
    "ConfigTuning": "ResourceMgmt",
    "PerformanceOptimization": "ResourceMgmt",
    "HealthAudit": "ResourceMgmt",

    # Security
    "ThreatMitigation": "SecurityOperation",
    "SecurityOperation": "SecurityOperation",
}

def normalize_task_type(task_type: Any) -> str:
    """
    Map diverse task type labels into our canonical vocabulary.
    - Accepts any input; non-strings pass through as their str() for safety.
    - Matches aliases case-insensitively.
    """
    if not isinstance(task_type, str):
        return str(task_type)

    s = task_type.strip()
    if not s:
        return s  # empty string stays empty

    # Exact alias match (case-sensitive)
    if s in TASK_TYPE_ALIASES:
        return TASK_TYPE_ALIASES[s]

    # Case-insensitive match
    sl = s.lower()
    for k, v in TASK_TYPE_ALIASES.items():
        if k.lower() == sl:
            return v

    # Already canonical or unknown—return as-is
    return s


# Role inference from typical agent names, e.g.:
#  - "ProtoAgent_Observer_instance_1"
#  - "Observer" (already the role)
#  - "Security-1" etc.
_ROLE_PATTERNS = [
    re.compile(r"^ProtoAgent_(?P<role>[A-Za-z]+)_", re.IGNORECASE),
    re.compile(r"^(?P<role>Observer|Security|Planner|Worker)\b", re.IGNORECASE),
]

# Optional explicit overrides (exact agent name → canonical role)
AGENT_ROLE_OVERRIDES: Dict[str, str] = {
    # "CVA-Observer": "Observer",
}

def infer_agent_role(agent_name: str, default: str = "Worker") -> str:
    """Best-effort mapping from agent instance name to canonical role key."""
    if not isinstance(agent_name, str) or not agent_name.strip():
        return default
    s = agent_name.strip()

    # 1) explicit override
    override = AGENT_ROLE_OVERRIDES.get(s)
    if override in ROLE_ALLOWED_TASKS:
        return override

    # 2) direct match to role key (case-insensitive)
    sl = s.lower()
    for role in ROLE_ALLOWED_TASKS.keys():
        if sl == role.lower():
            return role

    # 3) pattern-based extraction
    for rx in _ROLE_PATTERNS:
        m = rx.search(s)
        if m:
            extracted = m.group("role")
            el = extracted.lower()
            for role in ROLE_ALLOWED_TASKS.keys():
                if role.lower() == el:
                    return role

    # 4) fallback
    return default

def validate_role_task_assignment(agent_or_role: str, task_type: str) -> bool:
    """
    Accepts an agent instance name OR a role name; returns True if the role
    is allowed to perform the (canonicalized) task type.
    """
    role = agent_or_role if agent_or_role in ROLE_ALLOWED_TASKS else infer_agent_role(agent_or_role)
    allowed = ROLE_ALLOWED_TASKS.get(role, set())
    canonical = normalize_task_type(task_type)
    return canonical in allowed

# Back-compat: some older code checks role vs tool directly.
def validate_role_tool_assignment(agent_or_role: str, tool: str) -> bool:
    # NOTE: make sure resolve_task_type(tool=...) is defined earlier in this module.
    task_type = resolve_task_type(tool=tool)
    return validate_role_task_assignment(agent_or_role, task_type)

# -------------------------------------------------------------------------
# Intent utils (synonyms, placeholders, normalization)
# -------------------------------------------------------------------------

# Conservative synonym map → canonical intent (keep this tight)
INTENT_SYNONYMS: Dict[str, str] = {
    "security": "security_audit",
    "perf": "performance_optimization",
    "performance": "performance_optimization",
    "workflow": "workflow_optimization",
    "memory": "memory_optimization",
    "config": "config_tuning",
    "configuration": "config_tuning",
    "health": "health_audit",
    "status": "status_reporting",
    "report": "status_reporting",
}

# Values treated as "no intent specified"
_PLACEHOLDER_STRINGS: Set[str] = {
    "", " ", "tbd", "todo", "none", "null", "n/a", "na",
    "unspecified", "no specific intent",
}

def normalize_intent(val: Optional[str]) -> Optional[str]:
    """
    Normalize an explicit step['intent'] value.
    - Returns canonical intent if valid
    - Maps conservative synonyms to canonical
    - Returns None for placeholders / unknowns / non-strings
    """
    if not isinstance(val, str):
        return None
    raw = val.strip().lower()
    if raw.replace('_','') == 'statusreporting':
        return 'status_reporting'
    if raw.replace('_','') == 'statusreporting':
        return 'status_reporting'
    if not raw or raw in _PLACEHOLDER_STRINGS:
        return None
    if raw in ALLOWED_INTENTS:
        return raw
    return INTENT_SYNONYMS.get(raw)

# -------------------------------------------------------------------------
# High-level helpers (diagnostics & tests)
# -------------------------------------------------------------------------

def explain_step_policy(step: dict) -> Dict[str, Any]:
    """
    Returns a structured explanation of how this step fares against policy.
    Useful for logs, tests, and UI.

    Keys:
      - intent_ok, agent_ok, tool_ok, role_ok
      - intent (resolved), intent_source
      - role (resolved), tool (canonical), task_type (resolved)
      - reasons (list of strings)
    """
    reasons: list[str] = []

    # Intent (single source of truth)
    intent_ok, intent_value, intent_reasons, intent_source = resolve_strategic_intent(
        step_intent=step.get("intent"),
        mission_type=step.get("mission_type"),
        tool=step.get("tool"),
        task_type_hint=step.get("task_type"),
        title=step.get("title"),
    )
    reasons.extend(intent_reasons)

    # Agent
    agent = (step.get("agent") or "").strip()
    agent_ok = bool(agent)
    if not agent_ok:
        reasons.append("Missing agent.")

    # Tool (ask the real registry, not local maps)
    tool_raw = step.get("tool") or step.get("tool_name") or ""
    tool_ok = _policy_has_tool(tool_raw)
    if not tool_ok:
        reasons.append(f"Unknown or unregistered tool '{tool_raw}'.")
    tool = _policy_canonical_tool(tool_raw) or tool_raw

    # Role & Task type
    role = infer_agent_role(agent)
    task_type = _policy_resolve_task_type(step)
    role_ok = validate_role_task_assignment(role, task_type)
    if not role_ok:
        reasons.append(f"Role '{role}' not permitted to run task_type '{task_type}'.")

    return {
        "intent_ok": intent_ok,
        "agent_ok": agent_ok,
        "tool_ok": tool_ok,
        "role_ok": role_ok,
        "intent": intent_value,
        "intent_source": intent_source,
        "role": role,
        "tool": tool,
        "task_type": task_type,
        "reasons": reasons,
    }

def policy_snapshot() -> Dict[str, Any]:
    """
    Inspectable snapshot for debugging, metrics, or admin UIs.
    """
    return {
        "allowed_intents": sorted(ALLOWED_INTENTS),
        "task_types": sorted(TASK_TYPES),
        "role_allowed_tasks": {k: sorted(v) for k, v in ROLE_ALLOWED_TASKS.items()},
        "tool_task_types_count": len(TOOL_TASK_TYPES),
        "has_aliases": bool(TOOL_ALIASES),
    }

# Initialize tool mappings on module load
_bootstrap_tool_mappings()
