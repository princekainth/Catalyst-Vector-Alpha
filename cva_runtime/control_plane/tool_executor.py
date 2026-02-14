"""Tool execution facade.

Phase 1 split: wraps global ToolRegistry safe_call without behavior changes.
"""

from __future__ import annotations

from typing import Any

from tool_registry import tool_registry


class ToolExecutor:
    def __init__(self):
        self.registry = tool_registry

    def run(self, tool_name: str, timeout_seconds: int | None = None, **kwargs: Any) -> Any:
        return self.registry.safe_call(tool_name, timeout_seconds=timeout_seconds, **kwargs)
