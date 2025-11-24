# core/smoke_normalizer.py
from types import SimpleNamespace
from pprint import pprint

import core.policy as policy
from tool_registry import tool_registry
import utils

AVAILABLE_AGENTS = {
    "ProtoAgent_Observer_instance_1",
    "ProtoAgent_Security_instance_1",
    "ProtoAgent_Worker_instance_1",
    "ProtoAgent_Planner_instance_1",
}

def explain_step(step, available_agents, available_tools):
    title = (step.get("title") or "").strip()
    agent = (step.get("agent") or "").strip()
    tool  = (step.get("tool") or "").strip()

    intent_ok = policy.validate_step_intent(step)
    agent_ok  = agent in available_agents
    tool_ok   = tool in available_tools

    # derive role safely
    parts = agent.split("_")
    agent_role = parts[1] if len(parts) >= 2 else "Worker"

    # resolve task type using step + tool (matches your policy signature)
    task_type = policy.resolve_task_type(step, tool) if hasattr(policy, "resolve_task_type") else "GenericTask"
    role_ok   = policy.validate_role_task_assignment(agent_role, task_type) if hasattr(policy, "validate_role_task_assignment") else True

    reasons = []
    if not intent_ok: reasons.append("bad_intent")
    if not agent_ok:  reasons.append("bad_agent")
    if not tool_ok:   reasons.append("bad_tool")
    if not role_ok:   reasons.append(f"role_mismatch (role={agent_role}, task={task_type})")

    return {
        "title": title, "agent": agent, "tool": tool,
        "intent_ok": intent_ok, "agent_ok": agent_ok, "tool_ok": tool_ok, "role_ok": role_ok,
        "reasons": reasons
    }

def main():
    available_tools = tool_registry.get_available_tools()

    # (optional) sanity: make sure every tool has a task type mapping
    if hasattr(policy, "TOOL_TO_TASK_TYPE"):
        missing = available_tools - set(policy.TOOL_TO_TASK_TYPE.keys())
        if missing:
            raise SystemExit(f"Policy mapping missing entries for tools: {sorted(missing)}")

    demo_plan = {
        "id": "plan-demo",
        "steps": [
            {
                "title": "Security audit: run a quick ping sweep",
                "intent": "security_audit",
                "agent": "ProtoAgent_Security_instance_1",
                "tool": "initiate_network_scan",
                "args": {"target_ip": "10.0.0.5", "scan_type": "ping_sweep"},
            },
            {
                "title": "no specific intent task",
                "agent": "ProtoAgent_Worker_instance_1",
                "tool": "web_search",
                "args": {"query": "stuff"},
            },
            {
                "title": "Read and then create report",
                "intent": "status_reporting",
                "agent": "ProtoAgent_Worker_instance_1",
                "tools": ["read_webpage", "create_pdf"],
                "args": {"url": "https://example.com"},
            },
            {
                "title": "Check CPU",
                "intent": "performance_optimization",
                "agent": "UnknownAgent",
                "tool": "get_system_cpu_load",
                "args": {"time_interval_seconds": 1, "samples": 3, "per_core": False},
            },
            {
                "title": "Planner executes tool",
                "intent": "workflow_optimization",
                "agent": "ProtoAgent_Planner_instance_1",
                "tool": "create_pdf",
                "args": {"filename": "Report", "text_content": "hi"},
            },
            {
                "title": "Status reporting: compile a brief PDF",
                "intent": "status_reporting",
                "agent": "ProtoAgent_Worker_instance_1",
                "tool": "create_pdf",
                "args": {"filename": "Daily_Status", "text_content": "All systems nominal."},
            },
        ]
    }

    # Run through your normalizer (self is unused so pass a dummy)
    clean, skips = utils._normalize_plan_schema(SimpleNamespace(), demo_plan, AVAILABLE_AGENTS, tool_registry.get_available_tools())

    print("\n=== Normalization Result ===")
    print("Clean steps:", len(clean))
    pprint(clean, width=100)
    print("Skips summary:", skips)

    print("\n=== Policy Explanations (per original step) ===")
    for s in demo_plan["steps"]:
        expl = explain_step(s, AVAILABLE_AGENTS, tool_registry.get_available_tools())
        ok = expl["intent_ok"] and expl["agent_ok"] and expl["tool_ok"] and expl["role_ok"]
        print(f"- {expl['title']}\n  agent={expl['agent']} tool={expl['tool']} ok={ok} reasons={expl['reasons']}")

if __name__ == "__main__":
    main()
