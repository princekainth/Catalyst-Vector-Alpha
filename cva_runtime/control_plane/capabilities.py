"""Control-plane capabilities and tool risk profiles.

Commit 1 scope: primitives only (no wiring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, Tuple


class Capability(str, Enum):
    K8S_READ = "k8s_read"
    K8S_WRITE = "k8s_write"
    SHELL_READ = "shell_read"
    SHELL_WRITE = "shell_write"
    NET_OUTBOUND = "net_outbound"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    METRICS_READ = "metrics_read"
    LOGS_READ = "logs_read"
    LLM_CALL = "llm_call"
    APPROVAL_OVERRIDE = "approval_override"


class ToolRisk(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class ToolProfile:
    name: str
    required_caps: FrozenSet[Capability] = field(default_factory=frozenset)
    risk: ToolRisk = ToolRisk.SAFE
    resources_touched: Tuple[str, ...] = ()


def _caps(*caps: Capability) -> FrozenSet[Capability]:
    return frozenset(caps)


# v1 default profiles. Unknown tools are intentionally not listed.
DEFAULT_TOOL_PROFILES: Dict[str, ToolProfile] = {
    # Read-path tools
    "kubernetes_pod_metrics": ToolProfile(
        name="kubernetes_pod_metrics",
        required_caps=_caps(Capability.K8S_READ, Capability.METRICS_READ),
        risk=ToolRisk.SAFE,
        resources_touched=("kubernetes", "metrics"),
    ),
    "get_pod_status": ToolProfile(
        name="get_pod_status",
        required_caps=_caps(Capability.K8S_READ),
        risk=ToolRisk.SAFE,
        resources_touched=("kubernetes",),
    ),
    "watch_k8s_events": ToolProfile(
        name="watch_k8s_events",
        required_caps=_caps(Capability.K8S_READ, Capability.LOGS_READ),
        risk=ToolRisk.CAUTION,
        resources_touched=("kubernetes", "logs"),
    ),
    "web_search": ToolProfile(
        name="web_search",
        required_caps=_caps(Capability.NET_OUTBOUND),
        risk=ToolRisk.CAUTION,
        resources_touched=("network",),
    ),
    "read_webpage": ToolProfile(
        name="read_webpage",
        required_caps=_caps(Capability.NET_OUTBOUND),
        risk=ToolRisk.CAUTION,
        resources_touched=("network",),
    ),
    # Destructive infra mutations (default gated)
    "k8s_scale": ToolProfile(
        name="k8s_scale",
        required_caps=_caps(Capability.K8S_WRITE),
        risk=ToolRisk.DESTRUCTIVE,
        resources_touched=("kubernetes", "deployment"),
    ),
    "k8s_scale_deployment": ToolProfile(
        name="k8s_scale_deployment",
        required_caps=_caps(Capability.K8S_WRITE),
        risk=ToolRisk.DESTRUCTIVE,
        resources_touched=("kubernetes", "deployment"),
    ),
    "restart_agent": ToolProfile(
        name="restart_agent",
        required_caps=_caps(Capability.SHELL_WRITE),
        risk=ToolRisk.DESTRUCTIVE,
        resources_touched=("runtime",),
    ),
    "microsoft_autonomous_remediation": ToolProfile(
        name="microsoft_autonomous_remediation",
        required_caps=_caps(Capability.K8S_WRITE),
        risk=ToolRisk.DESTRUCTIVE,
        resources_touched=("kubernetes",),
    ),
    "deploy_recovery_protocol": ToolProfile(
        name="deploy_recovery_protocol",
        required_caps=_caps(Capability.K8S_WRITE, Capability.SHELL_WRITE),
        risk=ToolRisk.DESTRUCTIVE,
        resources_touched=("kubernetes", "runtime"),
    ),
    "execute_terminal_command": ToolProfile(
        name="execute_terminal_command",
        required_caps=_caps(Capability.SHELL_WRITE),
        risk=ToolRisk.DESTRUCTIVE,
        resources_touched=("runtime", "filesystem"),
    ),
    "write_sandbox_file": ToolProfile(
        name="write_sandbox_file",
        required_caps=_caps(Capability.FILE_WRITE),
        risk=ToolRisk.DESTRUCTIVE,
        resources_touched=("filesystem",),
    ),
}


def get_tool_profile(tool_name: str, profiles: Dict[str, ToolProfile] | None = None) -> ToolProfile | None:
    table = profiles or DEFAULT_TOOL_PROFILES
    return table.get((tool_name or "").strip())


def merge_tool_profiles(*tables: Dict[str, ToolProfile]) -> Dict[str, ToolProfile]:
    merged: Dict[str, ToolProfile] = {}
    for table in tables:
        merged.update(table)
    return merged


def capabilities_set(values: Iterable[Capability]) -> FrozenSet[Capability]:
    return frozenset(values)
