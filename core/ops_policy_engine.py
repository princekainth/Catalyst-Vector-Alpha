from __future__ import annotations
import time, logging
from typing import Dict, Any, Optional, List

class OpsPolicyEngine:
    """
    Explicit, auditable threshold policy with approval, cooldown, and change budgets.
    Kept separate from core/policy.py to avoid collisions.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger("OpsPolicyEngine")
        self._last_action_times: Dict[str, float] = {}
        self._action_counts: Dict[str, List[float]] = {}

    def eval_threshold_rule(self, rule: Dict[str, Any], metric_value: float, now: Optional[float] = None) -> Dict[str, Any]:
        now = now or time.time()
        rid = rule["id"]
        op = rule.get("op", ">")
        thr = float(rule["threshold"])
        approval = rule.get("approval", "human")
        cooldown = int(rule.get("cooldown_seconds", 300))
        budget = int(rule.get("change_budget_per_hour", 1))
        def trig(v: float) -> bool: return (v > thr) if op == ">" else (v < thr)

        last = self._last_action_times.get(rid, 0.0)
        if now - last < cooldown:
            return {"allow": False, "reason": f"cooldown {cooldown}s", "needs_approval": False}

        window = 3600
        recent = [t for t in self._action_counts.get(rid, []) if now - t < window]
        if len(recent) >= budget:
            return {"allow": False, "reason": f"budget exhausted ({budget}/h)", "needs_approval": False}

        if not trig(metric_value):
            return {"allow": False, "reason": "threshold not met", "needs_approval": False}

        return {"allow": True, "needs_approval": (approval != "auto"), "reason": "threshold triggered"}

    def record_action(self, rule_id: str, when: Optional[float] = None):
        when = when or time.time()
        self._last_action_times[rule_id] = when
        self._action_counts.setdefault(rule_id, []).append(when)
