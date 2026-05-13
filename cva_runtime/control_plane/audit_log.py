"""Append-only audit log primitives for control-plane decisions/actions.

Commit 1 scope: standalone logger utility (no wiring).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Mapping


REDACT_KEYS = {
    "token",
    "password",
    "passwd",
    "secret",
    "key",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "credentials",
    "env_value",
    "value",
    "env_val",
}


@dataclass(frozen=True)
class AuditRecord:
    timestamp: float
    trace_id: str
    agent_id: str
    tool: str
    args_hash: str
    decision: str
    reason: str
    result_status: str
    latency_ms: int


def _audit_path() -> str:
    return os.getenv("CVA_AUDIT_LOG_PATH", "./.cva/audit/actions.jsonl")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k)
            if key.lower() in REDACT_KEYS:
                out[key] = "***REDACTED***"
            else:
                out[key] = _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def hash_args(args: Mapping[str, Any] | None) -> str:
    safe_args = _redact(dict(args or {}))
    canonical = json.dumps(safe_args, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def append_record(record: AuditRecord, *, extra: Mapping[str, Any] | None = None) -> None:
    path = _audit_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload: dict[str, Any] = asdict(record)
    if extra:
        payload["extra"] = _redact(dict(extra))

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_decision(
    *,
    trace_id: str,
    agent_id: str,
    tool: str,
    args: Mapping[str, Any] | None,
    decision: str,
    reason: str,
    result_status: str,
    latency_ms: int,
    extra: Mapping[str, Any] | None = None,
) -> None:
    record = AuditRecord(
        timestamp=time.time(),
        trace_id=trace_id,
        agent_id=agent_id,
        tool=tool,
        args_hash=hash_args(args),
        decision=decision,
        reason=reason,
        result_status=result_status,
        latency_ms=max(0, int(latency_ms)),
    )
    append_record(record, extra=extra)
