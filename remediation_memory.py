import time
from typing import Any, Callable, Dict, List, Optional, Sequence


class RemediationMemory:
    """
    Lightweight in-memory store for remediation incidents.
    Each record: incident_type, forensics, fix_applied, outcome, timestamp.
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def record(
        self,
        incident_type: str,
        forensics: Dict[str, Any],
        fix_applied: str,
        outcome: str,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        entry = {
            "incident_type": incident_type,
            "forensics": forensics or {},
            "fix_applied": fix_applied,
            "outcome": outcome,
            "timestamp": timestamp if timestamp is not None else time.time(),
        }
        self._records.append(entry)
        return entry

    def query_similar(
        self,
        incident_type: Optional[str] = None,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        def _match(rec: Dict[str, Any]) -> bool:
            if predicate and not predicate(rec):
                return False
            if incident_type and rec.get("incident_type") != incident_type:
                return False
            return True

        results = [r for r in self._records if _match(r)]
        results.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
        return results[:limit] if limit else results

    def success_rates_by_fix(self, window: Optional[int] = None) -> Dict[str, float]:
        """
        Compute success rate per fix strategy. Optionally limit to last N records.
        """
        records: Sequence[Dict[str, Any]] = self._records
        if window is not None and window > 0:
            records = records[-window:]

        tally: Dict[str, Dict[str, int]] = {}
        for rec in records:
            fix = rec.get("fix_applied", "unknown")
            outcome = rec.get("outcome", "").lower()
            if fix not in tally:
                tally[fix] = {"success": 0, "total": 0}
            tally[fix]["total"] += 1
            if outcome == "success":
                tally[fix]["success"] += 1

        rates: Dict[str, float] = {}
        for fix, counts in tally.items():
            total = counts["total"]
            success = counts["success"]
            rates[fix] = success / total if total else 0.0
        return rates
