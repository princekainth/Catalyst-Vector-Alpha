# rewards.py
"""
Reward scoring for Catalyst Vector Alpha.

- Uses richer signals when available:
    open_time_ms (avg), responsive_rate [0..1], cpu_pct (avg %), mem_pct (avg %)
- Falls back to your previous fields if needed:
    cpu_p95, throttled
- Scores are clipped to [0, 1]; higher is better.
- Tunable via environment variables:
    CVA_TARGET_OPEN_MS, CVA_W_OPEN, CVA_W_RESP, CVA_W_CPU, CVA_W_MEM
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os


# -------- Tunables (env overrides allowed) ------------------------------------
TARGET_OPEN_MS = float(os.getenv("CVA_TARGET_OPEN_MS", "10.0"))
W_OPEN         = float(os.getenv("CVA_W_OPEN", "0.80"))
W_RESP         = float(os.getenv("CVA_W_RESP", "0.10"))
W_CPU          = float(os.getenv("CVA_W_CPU",  "0.05"))
W_MEM          = float(os.getenv("CVA_W_MEM",  "0.05"))
# Optional throttle penalty (kept from your original behavior)
THROTTLE_PENALTY = float(os.getenv("CVA_THROTTLE_PENALTY", "-0.5"))
# -----------------------------------------------------------------------------


@dataclass
class RewardResult:
    score: float
    details: Dict[str, Any]


class RewardProvider:
    def score(self, mission: str, results: Dict[str, Any]) -> RewardResult:
        raise NotImplementedError


# ---------------------- Helpers ----------------------------------------------

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _norm_open(open_ms: Optional[float]) -> float:
    """
    1.0 at/below TARGET_OPEN_MS, then linearly decays to 0.0 at 2x target.
    None -> 0.5 (neutral).
    """
    if open_ms is None:
        return 0.5
    try:
        ratio = max(0.0, min(2.0, float(open_ms) / float(TARGET_OPEN_MS)))
    except Exception:
        return 0.5
    return 1.0 if ratio <= 1.0 else max(0.0, 1.0 - (ratio - 1.0))

def _norm_pct(pct: Optional[float]) -> float:
    """
    0% load => 1.0 (best), 100% => 0.0 (worst). None -> 0.5 (neutral).
    """
    if pct is None:
        return 0.5
    try:
        return max(0.0, 1.0 - (float(pct) / 100.0))
    except Exception:
        return 0.5
# -----------------------------------------------------------------------------


class WorkstationCaretakerReward(RewardProvider):
    def score(self, mission: str, results: Dict[str, Any]) -> RewardResult:
        """
        Inputs it considers (any may be missing):
          - open_time_ms (avg over tasks)
          - responsive_rate [0..1]
          - cpu_pct (avg %) or fallback: cpu_p95
          - mem_pct (avg %)
          - throttled (bool) optional penalty

        Returns: RewardResult with score in [0, 1] and debug details.
        """
        # --- pull metrics with safe fallbacks ---
        open_time_ms     = results.get("open_time_ms", results.get("open_time_ms_avg"))
        responsive_rate  = results.get("responsive_rate", None)

        cpu_pct = results.get("cpu_pct", None)
        if cpu_pct is None:
            cpu_p95 = results.get("cpu_p95", None)
            cpu_pct = float(cpu_p95) if cpu_p95 is not None else None

        mem_pct   = results.get("mem_pct", None)
        throttled = bool(results.get("throttled", False))

        # --- normalize to [0, 1] components ---
        f_open = _norm_open(open_time_ms)
        f_resp = responsive_rate if isinstance(responsive_rate, (int, float)) else 0.5
        f_cpu  = _norm_pct(cpu_pct)
        f_mem  = _norm_pct(mem_pct)

        # weighted sum + optional throttle penalty
        raw_score = (W_OPEN * f_open) + (W_RESP * f_resp) + (W_CPU * f_cpu) + (W_MEM * f_mem)
        if throttled:
            raw_score += THROTTLE_PENALTY

        score = _clip01(raw_score)

        return RewardResult(
            score=round(score, 3),
            details={
                # raw inputs
                "open_time_ms": open_time_ms,
                "responsive_rate": responsive_rate,
                "cpu_pct": cpu_pct,
                "mem_pct": mem_pct,
                "throttled": throttled,
                # normalized components
                "f_open": round(f_open, 3),
                "f_resp": round(f_resp, 3),
                "f_cpu": round(f_cpu, 3),
                "f_mem": round(f_mem, 3),
                # weights and targets (for debugging/tuning)
                "weights": {"open": W_OPEN, "resp": W_RESP, "cpu": W_CPU, "mem": W_MEM},
                "target_open_ms": TARGET_OPEN_MS,
            },
        )


REWARDS: Dict[str, RewardProvider] = {
    "workstation_caretaker": WorkstationCaretakerReward(),
}


# Convenience wrapper (optional)
def compute_reward(mission: str, results: Dict[str, Any], provider: str = "workstation_caretaker") -> RewardResult:
    rp = REWARDS.get(provider, REWARDS["workstation_caretaker"])
    return rp.score(mission, results)
