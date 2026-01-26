"""
SkillRegistry - Skill Crystallization Layer for CVA
Version 1.0: Turns repeated successful patterns into reusable skills.

Architecture:
- SkillCandidate detection from interest_history.jsonl
- Skill storage in .cva/skills.json
- Usage logging in .cva/skills_usage.jsonl
- Matching engine for Planner integration
- Degradation rules for stale skills
"""
import os
import json
import hashlib
import time
import math
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

# Constants
DEFAULT_CACHE_DIR = ".cva"
SKILLS_FILE = "skills.json"
SKILLS_USAGE_FILE = "skills_usage.jsonl"
PRIMITIVES_FILE = "primitives.jsonl"
INTEREST_HISTORY_FILE = "interest_history.jsonl"

# Thresholds
CANDIDATE_MIN_COUNT = 3
CANDIDATE_MIN_DISTINCT_TOPICS = 2
CANDIDATE_MIN_OUTCOME_SCORE = 0.70
CANDIDATE_WINDOW_DAYS = 14
SKILL_MATCH_THRESHOLD = 0.5
SKILL_DEGRADATION_FAILURES = 2
SKILL_DEGRADATION_WINDOW = 5

# Action verb normalization map
ACTION_VERBS = {
    "analyze": "analyze", "check": "analyze", "inspect": "analyze", "review": "analyze",
    "query": "query", "get": "query", "fetch": "query", "read": "query", "list": "query",
    "patch": "patch", "update": "patch", "modify": "patch", "fix": "patch",
    "scale": "scale", "resize": "scale", "adjust": "scale",
    "rollback": "rollback", "revert": "rollback", "undo": "rollback",
    "restart": "restart", "reboot": "restart", "reset": "restart",
    "deploy": "deploy", "create": "deploy", "apply": "deploy",
    "delete": "delete", "remove": "delete", "destroy": "delete",
    "monitor": "monitor", "watch": "monitor", "observe": "monitor",
}

# Primitive pattern lookup: (verb, tool_category) -> pattern
# This maps action+tool combinations to semantic primitives
PRIMITIVE_PATTERNS = {
    # Query patterns
    ("query", "prometheus"): "metric_fetch",
    ("query", "kubectl"): "state_inspection",
    ("query", "database"): "data_retrieval",
    ("query", "api"): "endpoint_query",
    # Analysis patterns
    ("analyze", "prometheus"): "metric_analysis",
    ("analyze", "kubectl"): "resource_analysis",
    ("analyze", "logs"): "log_analysis",
    # Mutation patterns
    ("patch", "kubectl"): "config_mutation",
    ("scale", "kubectl"): "replica_adjustment",
    ("restart", "kubectl"): "pod_lifecycle",
    ("deploy", "kubectl"): "workload_deployment",
    ("delete", "kubectl"): "resource_removal",
    ("rollback", "kubectl"): "version_revert",
    # Monitoring patterns
    ("monitor", "prometheus"): "metric_watch",
    ("monitor", "kubectl"): "resource_watch",
    # Generic fallbacks
    ("generic", "kubectl"): "k8s_operation",
    ("generic", "prometheus"): "metric_operation",
}

logger = logging.getLogger("CatalystLogger")


def _normalize_action_verb(action_text: str) -> str:
    """Extract and normalize the primary action verb from an action string."""
    words = action_text.lower().split()
    for word in words:
        clean = word.strip(".,;:()[]")
        if clean in ACTION_VERBS:
            return ACTION_VERBS[clean]
    return "generic"


def _compute_skill_signature(
    category: str,
    required_tools: List[str],
    action_pattern: str
) -> str:
    """Create a deterministic hash signature for skill matching."""
    normalized = f"{category}|{'|'.join(sorted(required_tools))}|{action_pattern}"
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


class Skill:
    """Represents a crystallized skill."""
    
    def __init__(
        self,
        skill_id: str,
        name: str,
        signature: str,
        category: str,
        required_tools: List[str],
        action_pattern: str,
        action_sequence: List[str],
        trigger_keywords: List[str],
        safety_flags: List[str],
        version: int = 1,
        success_rate: float = 1.0,
        times_used: int = 0,
        last_outcomes: Optional[List[bool]] = None,
        created_at: Optional[float] = None,
        last_used: Optional[float] = None,
        quarantined: bool = False,
    ):
        self.id = skill_id
        self.name = name
        self.signature = signature
        self.category = category
        self.required_tools = required_tools
        self.action_pattern = action_pattern
        self.action_sequence = action_sequence
        self.trigger_keywords = trigger_keywords
        self.safety_flags = safety_flags
        self.version = version
        self.success_rate = success_rate
        self.times_used = times_used
        self.last_outcomes = last_outcomes or []
        self.created_at = created_at or time.time()
        self.last_used = last_used
        self.quarantined = quarantined
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "signature": self.signature,
            "category": self.category,
            "required_tools": self.required_tools,
            "action_pattern": self.action_pattern,
            "action_sequence": self.action_sequence,
            "trigger_keywords": self.trigger_keywords,
            "safety_flags": self.safety_flags,
            "version": self.version,
            "confidence_model": {
                "success_rate": self.success_rate,
                "times_used": self.times_used,
                "last_outcomes": self.last_outcomes[-SKILL_DEGRADATION_WINDOW:],
            },
            "created_at": self.created_at,
            "last_used": self.last_used,
            "quarantined": self.quarantined,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        conf = data.get("confidence_model", {})
        return cls(
            skill_id=data["id"],
            name=data["name"],
            signature=data["signature"],
            category=data["category"],
            required_tools=data.get("required_tools", []),
            action_pattern=data.get("action_pattern", "generic"),
            action_sequence=data.get("action_sequence", []),
            trigger_keywords=data.get("trigger_keywords", []),
            safety_flags=data.get("safety_flags", []),
            version=data.get("version", 1),
            success_rate=conf.get("success_rate", 1.0),
            times_used=conf.get("times_used", 0),
            last_outcomes=conf.get("last_outcomes", []),
            created_at=data.get("created_at"),
            last_used=data.get("last_used"),
            quarantined=data.get("quarantined", False),
        )
    
    def record_outcome(self, success: bool) -> None:
        """Record a usage outcome and update confidence."""
        self.times_used += 1
        self.last_used = time.time()
        self.last_outcomes.append(success)
        
        # Keep only last N outcomes
        if len(self.last_outcomes) > SKILL_DEGRADATION_WINDOW:
            self.last_outcomes = self.last_outcomes[-SKILL_DEGRADATION_WINDOW:]
        
        # Update success rate (weighted towards recent)
        if self.last_outcomes:
            recent_weight = 0.7
            recent_rate = sum(self.last_outcomes) / len(self.last_outcomes)
            self.success_rate = (recent_weight * recent_rate) + ((1 - recent_weight) * self.success_rate)
        
        # Check for degradation
        recent_failures = self.last_outcomes[-SKILL_DEGRADATION_WINDOW:].count(False)
        if recent_failures >= SKILL_DEGRADATION_FAILURES:
            self.quarantined = True
            logger.warning(f"[SkillRegistry] ⚠️ Skill '{self.name}' quarantined due to {recent_failures} recent failures")


class SkillRegistry:
    """
    Registry for crystallized skills.
    - Detects candidates from interest history
    - Stores skills with versioning
    - Matches skills to contexts
    - Enforces safety gates
    """
    
    def __init__(self, repo_root: str, domain_loader=None):
        self.repo_root = repo_root
        self.cache_dir = os.path.join(repo_root, DEFAULT_CACHE_DIR)
        self.skills_path = os.path.join(self.cache_dir, SKILLS_FILE)
        self.usage_path = os.path.join(self.cache_dir, SKILLS_USAGE_FILE)
        self.history_path = os.path.join(self.cache_dir, INTEREST_HISTORY_FILE)
        self.domain_loader = domain_loader
        
        self._skills: Dict[str, Skill] = {}
        self.primitives_path = os.path.join(self.cache_dir, PRIMITIVES_FILE)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_skills()
    
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def _load_skills(self) -> None:
        """Load skills from disk."""
        if os.path.exists(self.skills_path):
            try:
                with open(self.skills_path, 'r') as f:
                    data = json.load(f)
                for skill_data in data.get("skills", []):
                    skill = Skill.from_dict(skill_data)
                    
                    # Loader-time validation: action_sequence must be List[dict]
                    if any(isinstance(act, str) for act in skill.action_sequence):
                        skill.quarantined = True
                        logger.warning(f"[SkillRegistry] ⚠️ Skill '{skill.name}' auto-quarantined: action_sequence contains strings")
                    
                    self._skills[skill.id] = skill
                logger.info(f"[SkillRegistry] Loaded {len(self._skills)} skills")
            except Exception as e:
                logger.error(f"[SkillRegistry] Failed to load skills: {e}")
    
    def _save_skills(self) -> None:
        """Save skills to disk."""
        try:
            data = {
                "version": 1,
                "updated_at": time.time(),
                "skills": [s.to_dict() for s in self._skills.values()]
            }
            with open(self.skills_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[SkillRegistry] Failed to save skills: {e}")
    
    def _log_usage(self, skill_id: str, success: bool, context: Dict[str, Any]) -> None:
        """Append usage to log."""
        entry = {
            "timestamp": time.time(),
            "skill_id": skill_id,
            "success": success,
            "context_summary": str(context)[:200]
        }
        try:
            with open(self.usage_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"[SkillRegistry] Failed to log usage: {e}")

    def _extract_primitives(self, skill: Skill) -> List[Dict[str, str]]:
        """
        Extract primitives from a skill's action sequence and tools.
        Returns list of {verb, target, pattern} dicts.
        """
        primitives = []
        
        for action in skill.action_sequence:
            # Handle both string and dict formats
            if isinstance(action, dict):
                # Extract intent from tool name (e.g., "get_pod_status" -> "get pod status")
                raw_text = action.get("tool", "") or action.get("description", "")
                # Replace underscores for better verb extraction splitting
                action_text = str(raw_text).replace("_", " ")
            else:
                action_text = str(action)

            verb = _normalize_action_verb(action_text)
            
            # Try to determine target from required tools
            for tool in skill.required_tools:
                tool_lower = tool.lower()
                
                # Categorize the tool
                if "kubectl" in tool_lower or "k8s" in tool_lower:
                    tool_category = "kubectl"
                elif "prometheus" in tool_lower or "prom" in tool_lower:
                    tool_category = "prometheus"
                elif "sql" in tool_lower or "postgres" in tool_lower or "db" in tool_lower:
                    tool_category = "database"
                elif "http" in tool_lower or "api" in tool_lower or "web" in tool_lower:
                    tool_category = "api"
                elif "log" in tool_lower:
                    tool_category = "logs"
                else:
                    tool_category = tool_lower
                
                # Look up pattern
                pattern = PRIMITIVE_PATTERNS.get(
                    (verb, tool_category),
                    PRIMITIVE_PATTERNS.get(("generic", tool_category), f"{verb}_{tool_category}")
                )
                
                primitives.append({
                    "verb": verb,
                    "target": tool_category,
                    "pattern": pattern
                })
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for p in primitives:
            key = (p["verb"], p["target"], p["pattern"])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        
        return unique

    def _log_primitives(
        self,
        skill_id: str,
        primitives: List[Dict[str, str]],
        success: Optional[bool] = None
    ) -> None:
        """
        Log extracted primitives to primitives.jsonl.
        This is the foundation for future transfer learning.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "primitives_detected": primitives,
            "success": success  # Will be None initially, updated on outcome
        }
        
        try:
            with open(self.primitives_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")
            logger.debug(f"[SkillRegistry] Logged {len(primitives)} primitives for {skill_id}")
        except Exception as e:
            logger.error(f"[SkillRegistry] Failed to log primitives: {e}")

    
    # -------------------------------------------------------------------------
    # Candidate Detection (Slice A)
    # -------------------------------------------------------------------------
    def detect_candidates(self) -> List[Dict[str, Any]]:
        """
        Scan interest_history.jsonl for skill candidates.
        Returns list of candidate signatures with their evidence.
        """
        if not os.path.exists(self.history_path):
            return []
        
        # Load history within window
        cutoff = time.time() - (CANDIDATE_WINDOW_DAYS * 86400)
        entries = []
        try:
            with open(self.history_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("timestamp", 0) >= cutoff:
                            entries.append(entry)
        except Exception as e:
            logger.error(f"[SkillRegistry] Failed to read history: {e}")
            return []
        
        # Group by signature
        # We need additional data not in current history - use what we have
        # For now, use topic as proxy for action pattern (to be enhanced later)
        signature_groups: Dict[str, List[Dict]] = defaultdict(list)
        
        for entry in entries:
            if entry.get("outcome_score", 0) < CANDIDATE_MIN_OUTCOME_SCORE:
                continue
            
            # Extract what we can - this will improve as we add more logging
            topic = entry.get("topic", "")
            category = entry.get("category", "generic")  # May not exist yet
            
            # Derive action pattern from notes or topic
            action_pattern = "generic"
            notes = entry.get("notes", "")
            for word in (notes + " " + topic).lower().split():
                if word in ACTION_VERBS:
                    action_pattern = ACTION_VERBS[word]
                    break
            
            # Create a loose signature for grouping
            sig_key = f"{category}|{action_pattern}"
            signature_groups[sig_key].append(entry)
        
        # Find candidates meeting threshold
        candidates = []
        for sig_key, group in signature_groups.items():
            if len(group) < CANDIDATE_MIN_COUNT:
                continue
            
            distinct_topics = len(set(e.get("topic", "") for e in group))
            if distinct_topics < CANDIDATE_MIN_DISTINCT_TOPICS:
                continue
            
            # This is a candidate!
            avg_score = sum(e.get("outcome_score", 0) for e in group) / len(group)
            candidates.append({
                "signature_key": sig_key,
                "count": len(group),
                "distinct_topics": distinct_topics,
                "avg_outcome_score": round(avg_score, 3),
                "sample_topics": list(set(e.get("topic", "") for e in group))[:3],
                "entries": group
            })
        
        logger.info(f"[SkillRegistry] Detected {len(candidates)} skill candidates")
        return candidates
    
    def promote_candidate(self, candidate: Dict[str, Any], name: str) -> Optional[Skill]:
        """
        Promote a candidate to a registered skill.
        """
        entries = candidate.get("entries", [])
        if not entries:
            return None
        
        # Extract common patterns
        sig_parts = candidate["signature_key"].split("|")
        category = sig_parts[0] if len(sig_parts) > 0 else "generic"
        action_pattern = sig_parts[1] if len(sig_parts) > 1 else "generic"
        
        # Collect tools from entries (if available)
        all_tools: Set[str] = set()
        all_keywords: Set[str] = set()
        for entry in entries:
            topic = entry.get("topic", "")
            all_keywords.update(topic.lower().split()[:5])
        
        signature = _compute_skill_signature(category, list(all_tools), action_pattern)
        
        # Check if skill with this signature already exists
        for existing in self._skills.values():
            if existing.signature == signature:
                logger.info(f"[SkillRegistry] Skill with signature {signature} already exists")
                return existing
        
        skill = Skill(
            skill_id=f"skill-{signature[:8]}-{int(time.time())}",
            name=name,
            signature=signature,
            category=category,
            required_tools=list(all_tools),
            action_pattern=action_pattern,
            action_sequence=[],  # To be populated from execution traces
            trigger_keywords=list(all_keywords)[:10],
            safety_flags=[],
            success_rate=candidate["avg_outcome_score"],
            times_used=candidate["count"],
        )
        
        self._skills[skill.id] = skill
        self._save_skills()
        logger.info(f"[SkillRegistry] 🎓 Promoted skill: {name} (sig: {signature[:8]})")
        return skill
    
    # -------------------------------------------------------------------------
    # Skill Registration (Slice B)
    # -------------------------------------------------------------------------
    def register_skill(
        self,
        name: str,
        category: str,
        required_tools: List[str],
        action_pattern: str,
        action_sequence: List[str],
        trigger_keywords: List[str],
        safety_flags: Optional[List[str]] = None,
    ) -> Skill:
        """Manually register a new skill."""
        signature = _compute_skill_signature(category, required_tools, action_pattern)
        
        skill = Skill(
            skill_id=f"skill-{signature[:8]}-{int(time.time())}",
            name=name,
            signature=signature,
            category=category,
            required_tools=required_tools,
            action_pattern=action_pattern,
            action_sequence=action_sequence,
            trigger_keywords=trigger_keywords,
            safety_flags=safety_flags or [],
        )
        
        self._skills[skill.id] = skill
        self._save_skills()
        logger.info(f"[SkillRegistry] Registered skill: {name}")
        return skill
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self._skills.get(skill_id)
    
    def list_skills(self) -> List[Skill]:
        """List all registered skills."""
        return list(self._skills.values())
    
    # -------------------------------------------------------------------------
    # Skill Matching (Slice B)
    # -------------------------------------------------------------------------
    def get_matching_skills(
        self,
        context: Dict[str, Any],
        available_tools: Optional[List[str]] = None,
        top_n: int = 3
    ) -> List[Tuple[Skill, float]]:
        """
        Find skills matching the given context.
        Returns list of (skill, score) tuples sorted by score descending.
        """
        if not self._skills:
            return []
        
        available_tools_set = set(t.lower() for t in (available_tools or []))
        context_category = context.get("category", "").lower()
        context_keywords = set(context.get("keywords", []))
        context_text = context.get("text", "").lower()
        
        matches = []
        for skill in self._skills.values():
            # Skip quarantined skills
            if skill.quarantined:
                continue
            
            # Category match (Prioritize specific matches, allow general_planning as fallback)
            if context_category and skill.category.lower() != context_category:
                if skill.category.lower() != "general_planning":
                    continue

            
            # Tool availability check
            if available_tools_set:
                required_set = set(t.lower() for t in skill.required_tools)
                if not required_set.issubset(available_tools_set):
                    continue
            
            # Compute trigger match strength
            trigger_match = 0.0
            if skill.trigger_keywords:
                keyword_overlap = len(context_keywords & set(skill.trigger_keywords))
                text_matches = sum(1 for kw in skill.trigger_keywords if kw in context_text)
                trigger_match = (keyword_overlap + text_matches) / len(skill.trigger_keywords)
            
            # Compute final score
            # Base score allows new skills to be selected; usage history boosts confidence
            # Formula: success_rate * (1 + log1p(times_used)) * trigger_match
            usage_boost = 1.0 + math.log1p(skill.times_used)
            score = skill.success_rate * usage_boost * max(0.1, trigger_match)
            
            if score > 0:
                matches.append((skill, score))

        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_n]
    
    # -------------------------------------------------------------------------
    # Skill Invocation (Slice C)
    # -------------------------------------------------------------------------
    def invoke_skill(
        self,
        skill_id: str,
        context: Dict[str, Any],
        domain_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke a skill, returning an action plan (not auto-executing).
        Enforces safety gates.
        """
        skill = self._skills.get(skill_id)
        if not skill:
            return {"status": "error", "error": f"Skill {skill_id} not found"}
        
        if skill.quarantined:
            return {
                "status": "blocked",
                "reason": "skill_quarantined",
                "message": f"Skill '{skill.name}' is quarantined due to recent failures. Requires validation."
            }
        
        # Safety gate check
        if skill.safety_flags and domain_context:
            risk_level = domain_context.get("domain", {}).get("risk_level", "low")
            approval_required = domain_context.get("domain", {}).get("requires_human_approval_for", [])
            
            needs_approval = any(flag in approval_required for flag in skill.safety_flags)
            if needs_approval and risk_level in ("medium", "high"):
                return {
                    "status": "requires_approval",
                    "skill_id": skill_id,
                    "skill_name": skill.name,
                    "safety_flags": skill.safety_flags,
                    "action_sequence": skill.action_sequence,
                    "message": f"Skill '{skill.name}' requires human approval for: {skill.safety_flags}"
                }
        
        # Return action plan (Planner will execute)
        result = {
            "status": "ready",
            "skill_id": skill_id,
            "skill_name": skill.name,
            "category": skill.category,
            "action_pattern": skill.action_pattern,
            "action_sequence": skill.action_sequence,
            "required_tools": skill.required_tools,
            "confidence": skill.success_rate,
            "times_used": skill.times_used,
        }
        
        # === PRIMITIVE EXTRACTION ===
        # Extract and log primitives for introspection and future transfer learning
        primitives = self._extract_primitives(skill)
        if primitives:
            result["primitives"] = primitives
            self._log_usage(skill_id, True, context) # Log intent-to-use
            self._log_primitives(skill_id, primitives, success=None)  # Success logged on outcome
            logger.info(f"[SkillRegistry] 🔬 Extracted {len(primitives)} primitives from '{skill.name}'")

        # === END PRIMITIVE EXTRACTION ===
        
        return result
    
    def record_skill_outcome(self, skill_id: str, success: bool, context: Dict[str, Any]) -> None:
        """Record the outcome of a skill invocation."""
        skill = self._skills.get(skill_id)
        if skill:
            skill.record_outcome(success)
            self._save_skills()
            self._log_usage(skill_id, success, context)
            
            # Backfill primitive success
            primitives = self._extract_primitives(skill)
            if primitives:
                self._log_primitives(skill_id, primitives, success=success)
    
    # -------------------------------------------------------------------------
    # Analytics
    # -------------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total = len(self._skills)
        active = sum(1 for s in self._skills.values() if not s.quarantined)
        quarantined = total - active
        
        if self._skills:
            avg_success = sum(s.success_rate for s in self._skills.values()) / total
            total_uses = sum(s.times_used for s in self._skills.values())
        else:
            avg_success = 0
            total_uses = 0
        
        return {
            "total_skills": total,
            "active_skills": active,
            "quarantined_skills": quarantined,
            "avg_success_rate": round(avg_success, 3),
            "total_invocations": total_uses,
        }
