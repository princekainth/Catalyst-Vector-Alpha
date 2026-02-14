"""Cognitive loop helpers.

Phase 1 split: wrapper around existing orchestrator loop entry.
"""

from __future__ import annotations

from typing import Any


def run_loop(orchestrator: Any, tick_sleep: int = 10) -> None:
    orchestrator.run_cognitive_loop(tick_sleep=tick_sleep)
