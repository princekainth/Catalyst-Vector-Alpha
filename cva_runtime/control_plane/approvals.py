"""Approval token issuance and validation with in-memory TTL cache."""

from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ApprovalGrant:
    token: str
    trace_id: str
    tool: str
    args_hash: str
    agent_id: Optional[str]
    issued_at: float
    expires_at: float
    used: bool = False


class ApprovalStore:
    def __init__(self, default_ttl_seconds: int = 300):
        self._default_ttl_seconds = max(1, int(default_ttl_seconds))
        self._lock = threading.Lock()
        self._tokens: Dict[str, ApprovalGrant] = {}

    def issue(
        self,
        *,
        trace_id: str,
        tool: str,
        args_hash: str,
        agent_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Tuple[str, int]:
        ttl = max(1, int(ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds))
        token = f"appr_{secrets.token_urlsafe(24)}"
        now = time.time()
        grant = ApprovalGrant(
            token=token,
            trace_id=str(trace_id),
            tool=str(tool),
            args_hash=str(args_hash),
            agent_id=(str(agent_id) if agent_id else None),
            issued_at=now,
            expires_at=now + ttl,
            used=False,
        )
        with self._lock:
            self._prune_locked(now)
            self._tokens[token] = grant
        return token, ttl

    def validate(
        self,
        *,
        token: str,
        trace_id: str,
        tool: str,
        args_hash: str,
        agent_id: Optional[str] = None,
        consume: bool = True,
    ) -> Tuple[bool, str]:
        if not token:
            return False, "missing approval token"

        now = time.time()
        with self._lock:
            self._prune_locked(now)
            grant = self._tokens.get(token)
            if not grant:
                return False, "approval token not found or expired"
            if grant.used:
                return False, "approval token already used"
            if now > grant.expires_at:
                self._tokens.pop(token, None)
                return False, "approval token expired"
            if grant.trace_id != str(trace_id):
                return False, "approval token trace mismatch"
            if grant.tool != str(tool):
                return False, "approval token tool mismatch"
            if grant.args_hash != str(args_hash):
                return False, "approval token args mismatch"
            if grant.agent_id and agent_id and grant.agent_id != str(agent_id):
                return False, "approval token agent mismatch"
            if consume:
                grant.used = True
        return True, "approval token valid"

    def _prune_locked(self, now: float) -> None:
        expired = [
            token
            for token, grant in self._tokens.items()
            if grant.used or now > grant.expires_at
        ]
        for token in expired:
            self._tokens.pop(token, None)


def _default_ttl() -> int:
    raw = os.getenv("CVA_APPROVAL_TTL_S", "300").strip()
    try:
        return max(1, int(raw))
    except Exception:
        return 300


_STORE = ApprovalStore(default_ttl_seconds=_default_ttl())


def issue_approval_token(
    *,
    trace_id: str,
    tool: str,
    args_hash: str,
    agent_id: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
) -> Tuple[str, int]:
    return _STORE.issue(
        trace_id=trace_id,
        tool=tool,
        args_hash=args_hash,
        agent_id=agent_id,
        ttl_seconds=ttl_seconds,
    )


def validate_approval_token(
    *,
    token: str,
    trace_id: str,
    tool: str,
    args_hash: str,
    agent_id: Optional[str] = None,
    consume: bool = True,
) -> Tuple[bool, str]:
    return _STORE.validate(
        token=token,
        trace_id=trace_id,
        tool=tool,
        args_hash=args_hash,
        agent_id=agent_id,
        consume=consume,
    )
