# core/mission_policy.py
from typing import List, Dict, Any
import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# ==============================================================================
# MISSION TOOL POLICIES
# ==============================================================================

MISSION_TOOL_POLICY = {
    "performance_optimization": {
        "allow": {
            "get_system_cpu_load",
            "get_system_resource_usage",
            "top_processes",
            "measure_responsiveness",
            "update_resource_allocation",
            "create_pdf",
            "generate_report_pdf",
            "k8s_scale",
            "k8s_restart"
        },
        "deny": {"analyze_threat_signature", "isolate_network_segment", "deploy_recovery_protocol"},
        "fallback": "update_resource_allocation",
    },
    
    "security_audit": {
        "allow": {
            "analyze_threat_signature", 
            "isolate_network_segment", 
            "deploy_recovery_protocol", 
            "create_pdf", 
            "generate_report_pdf",
            "k8s_scale",
            "k8s_restart"
        },
        "deny": set(),
        "fallback": "analyze_threat_signature",
    },
    
    "health_audit": {
        "allow": {
            "get_system_cpu_load",
            "get_system_resource_usage",
            "top_processes",
            "measure_responsiveness",
            "create_pdf",
            "generate_report_pdf",
        },
        "deny": {
            "analyze_threat_signature", 
            "isolate_network_segment", 
            "deploy_recovery_protocol", 
            "update_resource_allocation",
            "k8s_scale",
            "k8s_restart"
        },
        "fallback": "measure_responsiveness",
    },
    
    # NEW: Sandbox inspection mission - terminal access only
    "sandbox_inspection": {
        "allow": {
            "execute_terminal_command",  # Primary tool for this mission
            "create_pdf",
            "generate_report_pdf",
        },
        "deny": {
            "k8s_scale",
            "k8s_restart",
            "analyze_threat_signature",
            "isolate_network_segment",
            "deploy_recovery_protocol",
            "update_resource_allocation",
        },
        "fallback": "execute_terminal_command",
    },
}

# ==============================================================================
# PLAN FILTERING
# ==============================================================================

def filter_plan_steps(mission: str, steps: List[Dict[str, Any]], target_deployment: str = None) -> List[Dict[str, Any]]:
    """Filter and auto-correct plan steps. Overrides deployment names if target_deployment provided."""
    policy = MISSION_TOOL_POLICY.get(mission, {"allow": set(), "deny": set(), "fallback": None})
    filtered_steps: List[Dict[str, Any]] = []

    for step in steps:
        tool = step.get("tool")
        if tool in policy["deny"] or (policy["allow"] and tool not in policy["allow"]):
            if policy["fallback"]:
                # Auto-correct the step to use the fallback tool
                step = dict(step)
                step["tool"] = policy["fallback"]
                step["title"] = f"[auto-corrected] {step.get('title', '')}".strip()
            else:
                # If no fallback, skip the step entirely
                continue
        
        # Override deployment name if target_deployment provided
        if target_deployment and tool in ["k8s_scale", "k8s_restart"]:
            step = dict(step)
            args = step.get("args", {})
            if isinstance(args, dict):
                if args.get("deployment") in ["nginx", None]:
                    args = dict(args)
                    args["deployment"] = target_deployment
                    step["args"] = args
                    print(f"[Policy] Overriding deployment: nginx → {target_deployment}")
        
        filtered_steps.append(step)
    return filtered_steps

def count_autocorrected(steps: List[Dict[str, Any]]) -> int:
    return sum(1 for s in steps if str(s.get("title", "")).startswith("[auto-corrected]"))

# ==============================================================================
# EPSILON-GREEDY MISSION SELECTOR
# ==============================================================================

# Tunables
_DEFAULT_EPS = float(os.getenv("CVA_EPSILON", "0.60"))   # 60% explore
_MIN_SAMPLES = int(os.getenv("CVA_MIN_SAMPLES", "3"))    # min outcomes before trusting avg
_LOOKBACK    = int(os.getenv("CVA_LOOKBACK", "500"))     # how many outcomes to scan

def candidate_missions() -> List[str]:
    """
    All available mission types that CVA can select from.
    """
    return [
        "performance_optimization",
        "security_audit",
        "scale_on_cpu_threshold",
        "general_planning",
        "health_audit",
        "sandbox_inspection",  # NEW: Terminal inspection mission
    ]

def _fetch_recent_outcomes(mem) -> List[Dict[str, Any]]:
    """
    Be tolerant of different mem kernels.
    Return a list of pure 'content' dicts.
    """
    # Try fetch_recent
    try:
        rows = mem.fetch_recent(mtype="MissionOutcome", limit=_LOOKBACK)
        out = []
        for r in rows:
            c = r["content"] if isinstance(r, dict) else json.loads(getattr(r, "content", "{}"))
            if isinstance(c, dict):
                out.append(c)
        return out
    except Exception:
        pass

    # Try get_recent_memories
    try:
        rows = mem.get_recent_memories("MissionOutcome", limit=_LOOKBACK)
        out = []
        for r in rows:
            c = r["content"] if isinstance(r, dict) else json.loads(getattr(r, "content", "{}"))
            if isinstance(c, dict):
                out.append(c)
        return out
    except Exception:
        pass

    # Try recent
    try:
        rows = mem.recent("MissionOutcome", limit=_LOOKBACK)
        out = []
        for r in rows:
            c = r.get("content", r)
            if not isinstance(c, dict):
                try:
                    c = json.loads(c)
                except Exception:
                    c = {}
            if isinstance(c, dict):
                out.append(c)
        return out
    except Exception:
        return []

def _score_table(mem) -> Tuple[Dict[str, float], Dict[str, int]]:
    sums, counts = defaultdict(float), defaultdict(int)
    for o in _fetch_recent_outcomes(mem):
        mission = o.get("mission") or "general_planning"
        score = o.get("score")
        if isinstance(score, (int, float)):
            sums[mission] += float(score)
            counts[mission] += 1
    avgs = {m: (sums[m] / counts[m]) for m in counts if counts[m] >= _MIN_SAMPLES}
    return avgs, counts

def select_next_mission(mem) -> str:
    """
    Pick next mission using epsilon-greedy:
      - exploit best average score most of the time
      - explore under-sampled missions some of the time
    """
    avgs, counts = _score_table(mem)
    cands = candidate_missions()

    if not avgs:
        choice = random.choice(cands)
        try:
            mem.logger.debug(f"[Planner] no history; exploring -> {choice}")
        except Exception:
            pass
        return choice

    # Best by average score
    best = max(avgs.items(), key=lambda kv: kv[1])[0]

    # Exploit vs explore
    if random.random() < (1.0 - _DEFAULT_EPS):
        try:
            mem.logger.debug(f"[Planner] exploit -> {best} (avg={avgs[best]:.3f}, n={counts.get(best,0)})")
        except Exception:
            pass
        return best

    # Explore: unseen preferred; otherwise least-sampled
    unseen = [m for m in cands if m not in avgs]
    if unseen:
        choice = random.choice(unseen)
    else:
        others = [(m, counts[m]) for m in avgs.keys() if m != best]
        others.sort(key=lambda x: x[1])
        choice = others[0][0] if others else best
    try:
        mem.logger.debug(f"[Planner] explore -> {choice}")
    except Exception:
        pass
    return choice
