"""Single control-plane choke point for tool execution."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, FrozenSet, Mapping

from cva_runtime.control_plane.audit_log import hash_args, log_decision
from cva_runtime.control_plane.capabilities import Capability, ToolRisk, get_tool_profile
from cva_runtime.control_plane.policy import evaluate


AGENT_CAPABILITY_PROFILES: Dict[str, FrozenSet[Capability]] = {
    "observer": frozenset(
        {
            Capability.K8S_READ,
            Capability.METRICS_READ,
            Capability.LOGS_READ,
            Capability.FILE_READ,
            Capability.LLM_CALL,
        }
    ),
    "planner": frozenset(
        {
            Capability.K8S_READ,
            Capability.METRICS_READ,
            Capability.LOGS_READ,
            Capability.FILE_READ,
            Capability.LLM_CALL,
        }
    ),
    "worker": frozenset(
        {
            Capability.K8S_READ,
            Capability.K8S_WRITE,
            Capability.METRICS_READ,
            Capability.LOGS_READ,
            Capability.FILE_READ,
            Capability.FILE_WRITE,
            Capability.SHELL_READ,
            Capability.SHELL_WRITE,
            Capability.LLM_CALL,
        }
    ),
    "security": frozenset(
        {
            Capability.K8S_READ,
            Capability.METRICS_READ,
            Capability.LOGS_READ,
            Capability.FILE_READ,
            Capability.LLM_CALL,
            Capability.APPROVAL_OVERRIDE,
        }
    ),
    "notifier": frozenset({Capability.NET_OUTBOUND, Capability.FILE_READ}),
}

FULL_CAPABILITIES: FrozenSet[Capability] = frozenset(Capability)


class ToolExecutor:
    def __init__(self, registry: Any):
        self.registry = registry

    def execute(
        self,
        agent_id: str,
        tool_name: str,
        args: Dict[str, Any],
        *,
        trace_id: str | None = None,
        context: Dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> Dict[str, Any]:
        started_ms = time.time()
        trace = (trace_id or self._trace_id()).strip()
        safe_args = dict(args or {})
        ctx = dict(context or {})
        warnings: list[str] = []

        profile = get_tool_profile(tool_name)
        if profile is None:
            reason = f"unknown tool '{tool_name}' is not in policy profiles"
            self._try_audit(
                trace_id=trace,
                agent_id=agent_id,
                tool_name=tool_name,
                args=safe_args,
                decision="POLICY_DECISION",
                reason=reason,
                result_status="deny",
                latency_ms=self._elapsed_ms(started_ms),
                extra={"allow": False, "requires_approval": False, "risk": ToolRisk.DESTRUCTIVE.value},
                warnings=warnings,
            )
            return self._error(
                code="policy_denied",
                message=reason,
                trace_id=trace,
                details={"tool": tool_name},
                warnings=warnings,
            )

        agent_caps = self._resolve_agent_capabilities(agent_id=agent_id, context=ctx)
        approved = bool(ctx.get("approved")) or bool(safe_args.get("approval_token"))
        decision = evaluate(
            agent_id=agent_id,
            tool_name=tool_name,
            args=safe_args,
            context=ctx,
            agent_capabilities=agent_caps,
            tool_profile=profile,
            approved=approved,
        )
        self._try_audit(
            trace_id=trace,
            agent_id=agent_id,
            tool_name=tool_name,
            args=safe_args,
            decision="POLICY_DECISION",
            reason=decision.reason,
            result_status=self._decision_status(decision.allow, decision.requires_approval),
            latency_ms=self._elapsed_ms(started_ms),
            extra={
                "allow": decision.allow,
                "requires_approval": decision.requires_approval,
                "risk": profile.risk.value,
                "required_caps": sorted(c.value for c in profile.required_caps),
            },
            warnings=warnings,
        )

        if not decision.allow:
            if decision.requires_approval:
                approval = {
                    "trace_id": trace,
                    "tool": tool_name,
                    "args_hash": hash_args(safe_args),
                    "approval_token": decision.approval_token,
                }
                return self._error(
                    code="approval_required",
                    message=decision.reason,
                    trace_id=trace,
                    approval=approval,
                    warnings=warnings,
                )
            return self._error(
                code="policy_denied",
                message=decision.reason,
                trace_id=trace,
                details={"tool": tool_name},
                warnings=warnings,
            )

        self._try_audit(
            trace_id=trace,
            agent_id=agent_id,
            tool_name=tool_name,
            args=safe_args,
            decision="TOOL_EXEC_START",
            reason="execution started",
            result_status="start",
            latency_ms=self._elapsed_ms(started_ms),
            extra={"risk": profile.risk.value},
            warnings=warnings,
        )

        exec_started = time.time()
        try:
            result = self.registry._safe_call_direct(
                tool_name,
                timeout_seconds=timeout_seconds,
                **safe_args,
            )
            status = self._result_status(result)
            self._try_audit(
                trace_id=trace,
                agent_id=agent_id,
                tool_name=tool_name,
                args=safe_args,
                decision="TOOL_EXEC_RESULT",
                reason="execution completed",
                result_status=status,
                latency_ms=self._elapsed_ms(exec_started),
                extra={"result_type": type(result).__name__},
                warnings=warnings,
            )
            return self._normalize_success_or_error(result=result, trace_id=trace, warnings=warnings)
        except Exception as exc:
            self._try_audit(
                trace_id=trace,
                agent_id=agent_id,
                tool_name=tool_name,
                args=safe_args,
                decision="TOOL_EXEC_RESULT",
                reason=f"execution exception: {exc}",
                result_status="error",
                latency_ms=self._elapsed_ms(exec_started),
                extra={"result_type": "exception"},
                warnings=warnings,
            )
            return self._error(
                code="tool_failed",
                message=str(exc),
                trace_id=trace,
                warnings=warnings,
            )

    def _resolve_agent_capabilities(
        self,
        *,
        agent_id: str,
        context: Mapping[str, Any],
    ) -> FrozenSet[Capability]:
        raw_caps = context.get("agent_capabilities")
        if isinstance(raw_caps, (list, tuple, set)):
            caps = []
            for cap in raw_caps:
                if isinstance(cap, Capability):
                    caps.append(cap)
                    continue
                try:
                    caps.append(Capability(str(cap).strip().lower()))
                except Exception:
                    continue
            if caps:
                return frozenset(caps)

        role_hint = str(context.get("agent_role") or context.get("agent_type") or agent_id or "").lower()
        for role, caps in AGENT_CAPABILITY_PROFILES.items():
            if role in role_hint:
                return caps

        # Compatibility fallback for legacy paths that do not pass agent metadata yet.
        return FULL_CAPABILITIES

    def _try_audit(
        self,
        *,
        trace_id: str,
        agent_id: str,
        tool_name: str,
        args: Mapping[str, Any],
        decision: str,
        reason: str,
        result_status: str,
        latency_ms: int,
        extra: Mapping[str, Any] | None,
        warnings: list[str],
    ) -> None:
        try:
            log_decision(
                trace_id=trace_id,
                agent_id=agent_id,
                tool=tool_name,
                args=args,
                decision=decision,
                reason=reason,
                result_status=result_status,
                latency_ms=latency_ms,
                extra=extra,
            )
        except Exception as exc:
            warnings.append(f"audit_log_failed:{exc}")

    def _normalize_success_or_error(self, *, result: Any, trace_id: str, warnings: list[str]) -> Dict[str, Any]:
        if isinstance(result, dict):
            payload = dict(result)
            payload.setdefault("status", "ok")
            if payload.get("status") == "ok":
                payload.setdefault("code", "ok")
            else:
                payload.setdefault("code", "tool_failed")
            payload.setdefault("error", None if payload.get("status") == "ok" else payload.get("message"))
            payload.setdefault("data", payload.get("result"))
            payload["trace_id"] = trace_id
            if warnings:
                payload["warnings"] = warnings
            return payload

        if isinstance(result, str) and result.startswith("[ERROR]"):
            return self._error(
                code="tool_failed",
                message=result,
                trace_id=trace_id,
                warnings=warnings,
            )

        payload = {
            "status": "ok",
            "code": "ok",
            "result": result,
            "data": result,
            "error": None,
            "summary": None,
            "trace_id": trace_id,
        }
        if warnings:
            payload["warnings"] = warnings
        return payload

    @staticmethod
    def _error(
        *,
        code: str,
        message: str,
        trace_id: str,
        details: Mapping[str, Any] | None = None,
        approval: Mapping[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": "error",
            "code": code,
            "message": message,
            "error": message,
            "data": None,
            "summary": None,
            "trace_id": trace_id,
        }
        if details:
            payload["details"] = dict(details)
        if approval:
            payload["approval"] = dict(approval)
        if warnings:
            payload["warnings"] = list(warnings)
        return payload

    @staticmethod
    def _trace_id() -> str:
        return f"trc_{uuid.uuid4().hex}"

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return max(0, int((time.time() - started_at) * 1000.0))

    @staticmethod
    def _result_status(result: Any) -> str:
        if isinstance(result, dict):
            return str(result.get("status", "ok"))
        if isinstance(result, str) and result.startswith("[ERROR]"):
            return "error"
        return "ok"

    @staticmethod
    def _decision_status(allow: bool, requires_approval: bool) -> str:
        if allow:
            return "allow"
        if requires_approval:
            return "approval_required"
        return "deny"
