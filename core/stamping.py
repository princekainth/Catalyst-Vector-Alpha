# -------------------------------------------------------------------------
# Step 2 — Stamp mission/intent/task on every planner step (DROP-IN)
# -------------------------------------------------------------------------

from typing import Dict, Any, List

# If you already defined these elsewhere, keep your originals.
INTENT_ENUM = {
    "SystemObservation",
    "EnvironmentObservation",
    "InformationRetrieval",
    "Planning",
    "StatusReporting",
    "Reporting",
    "ResourceTuning",
    "SecurityOperation",
    "ThreatMitigation",
    "PerformanceOptimization",
    "ConfigTuning",
    "HealthAudit",
}

# Tool → task_type mapping used to auto-derive step semantics
TOOL_TASK_TYPE: Dict[str, str] = {
    "create_pdf": "Reporting",
    "get_system_cpu_load": "SystemObservation",
    "get_system_resource_usage": "SystemObservation",
    "get_environmental_data": "EnvironmentObservation",
    "read_webpage": "InformationRetrieval",
    "web_search": "InformationRetrieval",
    "update_resource_allocation": "ResourceTuning",
    "analyze_threat_signature": "SecurityOperation",
    "isolate_network_segment": "ThreatMitigation",
}

def derive_task_type(tool: str) -> str:
    if not isinstance(tool, str):
        return "Reporting"  # safe default
    t = tool.strip()
    return TOOL_TASK_TYPE.get(t, "ToolRun")  # falls back to generic tool execution

def _coerce_intent(value: str, fallback: str) -> str:
    """
    Ensure value is a known intent; otherwise use fallback (if valid) or best-effort.
    """
    if isinstance(value, str) and value in INTENT_ENUM:
        return value
    if isinstance(fallback, str) and fallback in INTENT_ENUM:
        return fallback
    # last resort: pick something sensible
    return "StatusReporting"

def stamp_step(step: Dict[str, Any], mission_hint: str = None) -> Dict[str, Any]:
    """
    Ensure each step has: mission_type, strategic_intent, intent, task_type.
    - task_type derives from 'tool' if missing
    - strategic_intent defaults to mission_type if missing
    - mission_type falls back to mission_hint if missing
    - intent mirrors strategic_intent (some validators look for this exact key)
    """
    s = dict(step)  # copy

    # normalize aliases (harmless if already set)
    agent = s.get("agent") or s.get("agent_name")
    if agent:
        s["agent"] = agent
    tool = s.get("tool") or s.get("tool_name")
    if tool:
        s["tool"] = tool

    # 1) task_type
    tool = s.get("tool", "")  # re-read after alias normalize
    task_type = s.get("task_type") or derive_task_type(tool)
    s["task_type"] = task_type

    # --- NEW: infer a sensible mission if missing and no mission_hint ---
    inferred_mission = None
    if not mission_hint and not s.get("mission_type"):
        t = (tool or "").lower()
        if t in {"disk_usage", "get_system_cpu_load", "get_system_resource_usage", "measure_responsiveness", "top_processes"}:
            inferred_mission = "health_audit"
        elif t in {"update_resource_allocation", "renice_process", "ionice_process"}:
            inferred_mission = "performance_optimization"
        elif t in {"analyze_threat_signature", "isolate_network_segment", "deploy_recovery_protocol"}:
            inferred_mission = "security_audit"

    # 2) mission_type
    mission_type = s.get("mission_type") or mission_hint or inferred_mission or "StatusReporting"
    mission_type = _coerce_intent(mission_type, mission_hint or inferred_mission or "StatusReporting")
    s["mission_type"] = mission_type

    # 3) strategic_intent (defaults to mission_type)
    strategic_intent = s.get("strategic_intent") or mission_type
    strategic_intent = _coerce_intent(strategic_intent, mission_type)
    s["strategic_intent"] = strategic_intent

    # 4) intent (validators check this exact key)
    s.setdefault("intent", strategic_intent)

    return s

def stamp_plan(plan: Dict[str, Any], mission_fallback: str = "StatusReporting") -> Dict[str, Any]:
    """
    Given a plan dict: {"summary": "...", "steps": [ ... ]}
    returns a new plan with all steps stamped with required fields.
    """
    if not isinstance(plan, dict):
        return plan
    summary = plan.get("summary", "")
    steps: List[Dict[str, Any]] = plan.get("steps", [])

    # Prefer an explicit mission from the plan if present; else fallback
    plan_mission = plan.get("mission_type") or mission_fallback
    plan_mission = _coerce_intent(plan_mission, mission_fallback)

    stamped_steps = [stamp_step(st, mission_hint=plan_mission) for st in steps]
    out = dict(plan)
    out["steps"] = stamped_steps
    # Optionally stamp mission at plan-level for downstream consumers
    out.setdefault("mission_type", plan_mission)
    out.setdefault("strategic_intent", plan_mission)
    return out

# ------------------------------ quick checks ------------------------------
if __name__ == "__main__":
    sample_plan = {
        "summary": "Standard status report",
        "steps": [
            {
                "id": "S1",
                "title": "Emit status PDF",
                "agent": "ProtoAgent_Worker_instance_1",
                "tool": "create_pdf",
                "args": {"filename": "system_status_2025-09-22", "text_content": "CPU ~2–4%, Mem ~22–23%."},
                "depends_on": []
                # no mission/intent/task set on purpose
            }
        ]
    }

    stamped = stamp_plan(sample_plan, mission_fallback="StatusReporting")
    s1 = stamped["steps"][0]
    assert s1["task_type"] == "Reporting"
    assert s1["mission_type"] in INTENT_ENUM
    assert s1["strategic_intent"] in INTENT_ENUM
