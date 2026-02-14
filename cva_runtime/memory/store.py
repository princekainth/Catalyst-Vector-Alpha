"""Memory facade.

Phase 1 split: wraps existing CVA database + memetic kernel access patterns.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from database import cva_db


class MemoryStore:
    def __init__(self):
        self.db = cva_db

    def get_recent_tasks(self, limit: int = 50):
        return self.db.get_recent_tasks(limit=limit)

    def load_system_state(self, key: str, default: Any = None) -> Any:
        return self.db.load_system_state(key, default=default)

    def save_system_state(self, key: str, value: Any) -> None:
        self.db.save_system_state(key, value)

    def task_stats(self) -> Dict[str, Any]:
        return self.db.get_task_stats()
