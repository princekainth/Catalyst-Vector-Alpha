"""Control-plane policy stubs.

Phase 1 split: minimal policy hooks. Enforcement remains in existing runtime logic.
"""

from __future__ import annotations

from typing import Any, Dict


def allow_action(action: str, context: Dict[str, Any] | None = None) -> bool:
    return True


def require_human_approval(action: str, context: Dict[str, Any] | None = None) -> bool:
    return False
