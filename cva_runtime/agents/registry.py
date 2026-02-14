"""Agent registry helpers for v1 runtime."""

from __future__ import annotations

from typing import Dict, Any


CORE_AGENT_NAMES = (
    "ProtoAgent_Planner_instance_1",
    "ProtoAgent_Observer_instance_1",
    "ProtoAgent_Security_instance_1",
    "ProtoAgent_Worker_instance_1",
    "ProtoAgent_Notifier_instance_1",
)


def snapshot_agents(orchestrator: Any) -> Dict[str, Any]:
    return dict(getattr(orchestrator, "agent_instances", {}) or {})
