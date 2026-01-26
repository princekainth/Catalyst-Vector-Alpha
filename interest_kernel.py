"""
InterestKernel - Self-Improving, Domain-Aware Curiosity Engine for CVA
Version 2.0: Adds quality scoring, outcome feedback, safety gates, and tool-grounding.
"""
import os
import json
import yaml
import time
import uuid
import hashlib
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Constants
DEFAULT_CACHE_DIR = ".cva"
CACHE_FILE = "interest_cache.json"
HISTORY_FILE = "interest_history.jsonl"
DOMAIN_FILE = "DOMAIN.yaml"
DEFAULT_TTL_MINUTES = 240

# Safety blocklist - interests containing these terms require human approval
UNSAFE_ACTION_KEYWORDS = {
    "exfiltrate", "bypass", "exploit", "steal", "hack", "phish",
    "delete production", "drop database", "rm -rf", "sudo rm",
    "disable security", "disable auth", "leak credentials"
}

logger = logging.getLogger("CatalystLogger")


class InterestKernel:
    """
    A self-improving curiosity engine that:
    - Derives interests from DOMAIN.yaml and environment
    - Caches results with intelligent invalidation
    - Learns from outcomes to improve future curiosity
    - Enforces safety gates based on domain risk level
    """

    def __init__(self, repo_root: str, llm_integration=None, tool_registry=None):
        self.repo_root = repo_root
        self.llm = llm_integration
        self.tool_registry = tool_registry
        self.cache_dir = os.path.join(repo_root, DEFAULT_CACHE_DIR)
        self.cache_path = os.path.join(self.cache_dir, CACHE_FILE)
        self.history_path = os.path.join(self.cache_dir, HISTORY_FILE)
        self.domain_path = os.path.join(repo_root, DOMAIN_FILE)

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Domain Loading
    # -------------------------------------------------------------------------
    def load_domain_yaml(self) -> Dict[str, Any]:
        """Load domain intent from DOMAIN.yaml or return default."""
        if os.path.exists(self.domain_path):
            try:
                with open(self.domain_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"[InterestKernel] Failed to load DOMAIN.yaml: {e}")

        # Default fallback domain profile
        return {
            "domain": {
                "name": "CVA_Default",
                "mission": "Autonomous operations and self-evolving intelligence.",
                "current_focus": ["stability", "learning", "safety"],
                "risk_level": "low"
            }
        }

    def _get_risk_level(self, domain_context: Dict[str, Any]) -> str:
        """Extract risk level from domain context."""
        return domain_context.get("domain", {}).get("risk_level", "low")

    def _get_approval_requirements(self, domain_context: Dict[str, Any]) -> List[str]:
        """Get list of action types requiring human approval."""
        return domain_context.get("domain", {}).get("requires_human_approval_for", [])

    # -------------------------------------------------------------------------
    # Hashing & Cache Management
    # -------------------------------------------------------------------------
    def _get_input_hash(self, domain_context: Dict[str, Any], tool_names: List[str]) -> str:
        """Create a hash of inputs to detect changes."""
        input_str = json.dumps(domain_context, sort_keys=True) + "|".join(sorted(tool_names))
        return hashlib.sha256(input_str.encode('utf-8')).hexdigest()

    # -------------------------------------------------------------------------
    # History & Outcome Feedback
    # -------------------------------------------------------------------------
    def _load_recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Load recent interest history for feedback loop."""
        if not os.path.exists(self.history_path):
            return []

        entries = []
        try:
            with open(self.history_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
            return entries[-limit:]
        except Exception as e:
            logger.warning(f"[InterestKernel] Failed to load history: {e}")
            return []

    def _get_successful_interests(self, history: List[Dict]) -> List[str]:
        """Extract topics that led to successful outcomes."""
        return [
            h.get("topic", "")
            for h in history
            if h.get("outcome_score", 0) >= 0.7
        ]

    def _get_failed_interests(self, history: List[Dict]) -> List[str]:
        """Extract topics that failed or were not actionable."""
        return [
            h.get("topic", "")
            for h in history
            if h.get("outcome_score", 1.0) < 0.3
        ]

    def record_outcome(
        self,
        interest_id: str,
        topic: str,
        impact_score: float = 0.0,
        actionability_score: float = 0.0,
        gap_produced: Optional[str] = None,
        notes: str = "",
        tools_used: Optional[List[str]] = None,
        actions_taken: Optional[List[str]] = None,
        success: bool = True,
        category: str = "generic",
    ) -> None:
        """
        Record the outcome of an interest after exploration.
        This feeds back into the next derivation cycle AND skill crystallization.
        """
        outcome_score = (impact_score * 0.6) + (actionability_score * 0.4)

        # Normalize actions to verb patterns
        normalized_actions = []
        if actions_taken:
            for action in actions_taken:
                words = action.lower().split()
                for word in words:
                    if word in {"analyze", "check", "inspect", "query", "get", "fetch",
                                "patch", "update", "scale", "restart", "deploy", "delete",
                                "monitor", "rollback"}:
                        normalized_actions.append(word)
                        break

        entry = {
            "id": interest_id,
            "topic": topic,
            "category": category,
            "timestamp": time.time(),
            "impact_score": impact_score,
            "actionability_score": actionability_score,
            "outcome_score": round(outcome_score, 3),
            "success": success,
            "gap_produced": gap_produced,
            "notes": notes,
            "tools_used": tools_used or [],
            "actions_taken": normalized_actions,
        }

        try:
            with open(self.history_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(f"[InterestKernel] 📊 Recorded outcome for '{topic}': {outcome_score:.2f}")
        except Exception as e:
            logger.error(f"[InterestKernel] Failed to record outcome: {e}")


    # -------------------------------------------------------------------------
    # Safety Gate
    # -------------------------------------------------------------------------
    def _apply_safety_gate(
        self,
        interests: List[Dict[str, Any]],
        domain_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter or flag interests based on safety constraints.
        - Rejects interests with unsafe keywords in next_actions
        - Flags interests requiring human approval based on safety_flags
        """
        risk_level = self._get_risk_level(domain_context)
        approval_required = self._get_approval_requirements(domain_context)

        safe_interests = []
        for interest in interests:
            actions_text = " ".join(interest.get("next_actions", [])).lower()
            safety_flags = interest.get("safety_flags", [])

            # Check for unsafe keywords
            is_unsafe = any(kw in actions_text for kw in UNSAFE_ACTION_KEYWORDS)
            if is_unsafe:
                logger.warning(f"[InterestKernel] 🚫 Rejected unsafe interest: {interest.get('topic')}")
                continue

            # Check if any safety_flags require approval
            needs_approval = any(flag in approval_required for flag in safety_flags)
            if needs_approval and risk_level in ("medium", "high"):
                interest["requires_human_approval"] = True
                logger.info(f"[InterestKernel] ⚠️ Flagged for approval: {interest.get('topic')}")

            safe_interests.append(interest)

        return safe_interests

    # -------------------------------------------------------------------------
    # Tool Grounding Validation
    # -------------------------------------------------------------------------
    def _validate_tool_grounding(
        self,
        interests: List[Dict[str, Any]],
        available_tools: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Reject interests whose required_tools are not available.
        """
        available_set = set(t.lower() for t in available_tools)
        grounded = []

        for interest in interests:
            required = interest.get("required_tools", [])
            if not required:
                # No specific tools required - allow
                grounded.append(interest)
                continue

            missing = [t for t in required if t.lower() not in available_set]
            if missing:
                logger.info(f"[InterestKernel] ⛔ Skipping ungrounded interest '{interest.get('topic')}' - missing: {missing}")
                continue

            grounded.append(interest)

        return grounded

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------
    def get_interests(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Main entry point: returns cached interests or derives new ones."""
        domain_context = self.load_domain_yaml()
        tool_names = []
        if self.tool_registry and hasattr(self.tool_registry, 'list_tool_names'):
            tool_names = self.tool_registry.list_tool_names()

        current_hash = self._get_input_hash(domain_context, tool_names)

        # 1. Check Cache
        if not force_refresh and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    cache_data = json.load(f)

                cached_hash = cache_data.get("hash")
                timestamp = cache_data.get("timestamp", 0)
                ttl = cache_data.get("refresh_after_minutes", DEFAULT_TTL_MINUTES) * 60

                if cached_hash == current_hash and (time.time() - timestamp) < ttl:
                    logger.info("[InterestKernel] ⚡ Using cached curiosity targets.")
                    return cache_data.get("result", {})
            except Exception as e:
                logger.warning(f"[InterestKernel] Cache read failed: {e}")

        # 2. Derive new interests
        if not self.llm:
            logger.error("[InterestKernel] No LLM provided, cannot derive interests.")
            return {"interests": []}

        logger.info("[InterestKernel] 🧠 Deriving today's curiosity targets via LLM...")
        try:
            result = self.derive_interests(domain_context, tool_names)

            # 3. Apply safety gate
            result["interests"] = self._apply_safety_gate(result["interests"], domain_context)

            # 4. Validate tool grounding
            result["interests"] = self._validate_tool_grounding(result["interests"], tool_names)

            # 5. Save to Cache
            cache_entry = {
                "hash": current_hash,
                "timestamp": time.time(),
                "refresh_after_minutes": result.get("refresh_after_minutes", DEFAULT_TTL_MINUTES),
                "result": result
            }
            with open(self.cache_path, 'w') as f:
                json.dump(cache_entry, f, indent=2)

            return result
        except Exception as e:
            logger.error(f"[InterestKernel] Failed to derive interests: {e}")
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r') as f:
                    return json.load(f).get("result", {"interests": []})
            return {"interests": []}

    # -------------------------------------------------------------------------
    # LLM Derivation
    # -------------------------------------------------------------------------
    def derive_interests(self, domain_context: Dict[str, Any], tool_names: List[str]) -> Dict[str, Any]:
        """Call LLM to generate structured interests with full schema."""

        # Load history for feedback
        history = self._load_recent_history()
        successful = self._get_successful_interests(history)
        failed = self._get_failed_interests(history)

        system_prompt = """You are InterestKernel for Catalyst Vector Alpha (CVA).
Output STRICT JSON only. No prose. Follow the schema exactly.
Each interest must have a unique UUID id."""

        user_prompt = f"""
Derive 8-12 curiosity targets for today based on the domain and environment.

DOMAIN CONTEXT:
{json.dumps(domain_context, indent=2)}

TOOLS AVAILABLE:
{', '.join(tool_names)}

PAST SUCCESSFUL INTERESTS (prefer similar):
{', '.join(successful[:5]) if successful else 'None yet'}

PAST FAILED INTERESTS (avoid similar):
{', '.join(failed[:5]) if failed else 'None yet'}

CONSTRAINTS:
- Exploit > Explore ratio (70/30): 70% should improve current reliability/tools
- Each interest MUST have required_tools that exist in TOOLS AVAILABLE
- category must be one of: architecture, memory, safety, tooling, evaluation
- safety_flags should list any risky action types: writes, deletions, deployments, external_calls

OUTPUT FORMAT (strict JSON):
{{
  "interests": [
    {{
      "id": "uuid-string",
      "topic": "string",
      "category": "architecture|memory|safety|tooling|evaluation",
      "why_now": "string",
      "questions": ["string", "string"],
      "next_actions": ["string", "string"],
      "required_tools": ["tool_name"],
      "safety_flags": ["writes", "deployments"],
      "priority": 0.0-1.0
    }}
  ],
  "do_not_chase": ["string"],
  "refresh_after_minutes": integer
}}
"""

        response_text = self.llm.generate_text(
            prompt=user_prompt,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            json_mode=True
        )

        try:
            result = json.loads(response_text)
            if "interests" not in result:
                raise ValueError("LLM response missing 'interests' key")

            # Ensure all interests have IDs
            for interest in result["interests"]:
                if not interest.get("id"):
                    interest["id"] = str(uuid.uuid4())
                # Initialize tracking fields
                interest["outcome_logged"] = False

            return result
        except Exception as e:
            logger.error(f"[InterestKernel] Failed to parse LLM response: {e}")
            raise

    # -------------------------------------------------------------------------
    # Analytics API
    # -------------------------------------------------------------------------
    def summarize_trends(self, days: int = 7) -> Dict[str, Any]:
        """Summarize curiosity trends over the past N days."""
        history = self._load_recent_history(limit=100)
        cutoff = time.time() - (days * 86400)

        recent = [h for h in history if h.get("timestamp", 0) >= cutoff]

        if not recent:
            return {"message": "No history in timeframe", "count": 0}

        avg_impact = sum(h.get("impact_score", 0) for h in recent) / len(recent)
        avg_actionability = sum(h.get("actionability_score", 0) for h in recent) / len(recent)
        avg_outcome = sum(h.get("outcome_score", 0) for h in recent) / len(recent)

        top_topics = sorted(recent, key=lambda x: x.get("outcome_score", 0), reverse=True)[:3]

        return {
            "days": days,
            "total_interests": len(recent),
            "avg_impact_score": round(avg_impact, 3),
            "avg_actionability_score": round(avg_actionability, 3),
            "avg_outcome_score": round(avg_outcome, 3),
            "top_performing_topics": [t.get("topic") for t in top_topics]
        }
