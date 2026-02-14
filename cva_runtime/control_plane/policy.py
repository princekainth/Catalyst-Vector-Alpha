"""Control-plane policy evaluation primitives.

Commit 1 scope: decision model and rules only (no wiring).
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Any, FrozenSet, Mapping

from cva_runtime.control_plane.capabilities import Capability, ToolProfile, ToolRisk, get_tool_profile


@dataclass(frozen=True)
class PolicyDecision:
    allow: bool
    reason: str
    requires_approval: bool
    approval_token: str | None = None


def _approval_mode_from_env(context: Mapping[str, Any] | None = None) -> str:
    if context and context.get("approval_mode"):
        return str(context["approval_mode"]).strip().lower()
    return os.getenv("CVA_APPROVAL_MODE", "manual").strip().lower()


def _allow_destructive_from_env(context: Mapping[str, Any] | None = None) -> bool:
    if context and "allow_destructive" in context:
        return bool(context.get("allow_destructive"))
    return os.getenv("CVA_ALLOW_DESTRUCTIVE", "0").strip() == "1"


def _new_approval_token() -> str:
    return f"apr_{secrets.token_urlsafe(18)}"


def evaluate(
    agent_id: str,
    tool_name: str,
    args: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None = None,
    *,
    agent_capabilities: FrozenSet[Capability] | None = None,
    tool_profile: ToolProfile | None = None,
    approved: bool = False,
) -> PolicyDecision:
    """Evaluate whether an agent may execute a tool action.

    Rules (v1):
    - deny if profile unknown
    - deny if capability mismatch
    - destructive tools require approval unless explicitly enabled by env mode
    - deny by default when policy cannot decide
    """

    if not agent_id or not tool_name:
        return PolicyDecision(
            allow=False,
            reason="missing agent_id or tool_name",
            requires_approval=False,
        )

    profile = tool_profile or get_tool_profile(tool_name)
    if profile is None:
        return PolicyDecision(
            allow=False,
            reason=f"no policy profile for tool '{tool_name}'",
            requires_approval=False,
        )

    caps = agent_capabilities or frozenset()
    missing = sorted(cap.value for cap in profile.required_caps if cap not in caps)
    if missing:
        return PolicyDecision(
            allow=False,
            reason=f"agent '{agent_id}' lacks capabilities: {', '.join(missing)}",
            requires_approval=False,
        )

    if profile.risk != ToolRisk.DESTRUCTIVE:
        return PolicyDecision(allow=True, reason="allowed", requires_approval=False)

    # Destructive tools: apply approval mode gates.
    mode = _approval_mode_from_env(context)
    allow_destructive = _allow_destructive_from_env(context)

    if approved:
        return PolicyDecision(allow=True, reason="approved token accepted", requires_approval=False)

    if mode == "manual":
        return PolicyDecision(
            allow=False,
            reason="destructive action requires manual approval",
            requires_approval=True,
            approval_token=_new_approval_token(),
        )

    if mode == "autonomous_safe":
        return PolicyDecision(
            allow=False,
            reason="destructive action denied in autonomous_safe mode",
            requires_approval=False,
        )

    if mode == "autonomous_full":
        if allow_destructive:
            return PolicyDecision(
                allow=True,
                reason="destructive action allowed by CVA_ALLOW_DESTRUCTIVE=1",
                requires_approval=False,
            )
        return PolicyDecision(
            allow=False,
            reason="destructive action gated; set CVA_ALLOW_DESTRUCTIVE=1 or provide approval",
            requires_approval=True,
            approval_token=_new_approval_token(),
        )

    return PolicyDecision(
        allow=False,
        reason=f"unknown approval mode '{mode}' (deny by default)",
        requires_approval=False,
    )
