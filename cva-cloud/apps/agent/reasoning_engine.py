"""
Reasoning Engine - Makes CVA's thinking visible.
Every decision gets a reasoning trace.
"""

import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ReasoningStep:
    """Single step in a reasoning chain."""
    step_type: str  # observe, analyze, recall, decide, act, verify
    content: str
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass 
class ReasoningTrace:
    """Complete reasoning chain for one decision."""
    trace_id: str
    agent_name: str
    trigger: str  # What started this reasoning
    goal: str
    steps: List[ReasoningStep] = field(default_factory=list)
    outcome: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    completed_at: Optional[str] = None
    
    def add_step(self, step_type: str, content: str, confidence: float = 1.0, evidence: List[str] = None):
        """Add a reasoning step."""
        step = ReasoningStep(
            step_type=step_type,
            content=content,
            confidence=confidence,
            evidence=evidence or []
        )
        self.steps.append(step)
        return self
    
    def observe(self, content: str, evidence: List[str] = None):
        """Record an observation."""
        return self.add_step("observe", content, evidence=evidence)
    
    def analyze(self, content: str, confidence: float = 1.0):
        """Record analysis."""
        return self.add_step("analyze", content, confidence=confidence)
    
    def recall(self, content: str, evidence: List[str] = None):
        """Record memory recall."""
        return self.add_step("recall", content, evidence=evidence)
    
    def decide(self, content: str, confidence: float = 1.0, evidence: List[str] = None):
        """Record a decision."""
        return self.add_step("decide", content, confidence=confidence, evidence=evidence)
    
    def act(self, content: str):
        """Record an action taken."""
        return self.add_step("act", content)
    
    def verify(self, content: str, success: bool = True):
        """Record verification of action."""
        return self.add_step("verify", content, confidence=1.0 if success else 0.0)
    
    def complete(self, outcome: str):
        """Mark reasoning complete."""
        self.outcome = outcome
        self.completed_at = datetime.utcnow().isoformat() + "Z"
        return self
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"═══ REASONING TRACE: {self.trace_id[:8]} ═══",
            f"Agent: {self.agent_name}",
            f"Trigger: {self.trigger}",
            f"Goal: {self.goal}",
            "─── Steps ───"
        ]
        
        icons = {
            "observe": "👁️",
            "analyze": "🧠", 
            "recall": "💭",
            "decide": "⚖️",
            "act": "🔧",
            "verify": "✓" 
        }
        
        for i, step in enumerate(self.steps, 1):
            icon = icons.get(step.step_type, "•")
            conf = f" ({step.confidence:.0%})" if step.confidence < 1.0 else ""
            lines.append(f"  {i}. {icon} [{step.step_type.upper()}]{conf}: {step.content}")
            if step.evidence:
                for ev in step.evidence[:2]:  # Max 2 evidence items
                    lines.append(f"      └─ {ev[:60]}...")
        
        lines.append("─────────────")
        lines.append(f"Outcome: {self.outcome or 'In Progress'}")
        
        return "\n".join(lines)


class ReasoningEngine:
    """
    Central reasoning tracker for CVA.
    Stores all reasoning traces for review.
    """
    
    def __init__(self, persist_path: str = "./persistence_data/reasoning"):
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.active_traces: Dict[str, ReasoningTrace] = {}
        self._trace_counter = 0
    
    def start_trace(self, agent_name: str, trigger: str, goal: str) -> ReasoningTrace:
        """Start a new reasoning trace."""
        self._trace_counter += 1
        trace_id = f"{agent_name}_{int(time.time())}_{self._trace_counter}"
        
        trace = ReasoningTrace(
            trace_id=trace_id,
            agent_name=agent_name,
            trigger=trigger,
            goal=goal
        )
        
        self.active_traces[trace_id] = trace
        print(f"\n🧠 [REASONING] Started: {agent_name} → {goal[:50]}...")
        
        return trace
    
    def complete_trace(self, trace: ReasoningTrace, outcome: str):
        """Complete and persist a reasoning trace."""
        trace.complete(outcome)
        
        # Print summary
        print(trace.summary())
        
        # Persist to file
        trace_file = self.persist_path / f"{trace.trace_id}.json"
        with open(trace_file, "w") as f:
            f.write(trace.to_json())
        
        # Also append to daily log
        daily_log = self.persist_path / f"reasoning_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
        with open(daily_log, "a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")
        
        # Remove from active
        if trace.trace_id in self.active_traces:
            del self.active_traces[trace.trace_id]
        
        return trace
    
    def get_recent_traces(self, limit: int = 10) -> List[Dict]:
        """Get recent reasoning traces."""
        daily_log = self.persist_path / f"reasoning_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
        
        if not daily_log.exists():
            return []
        
        traces = []
        with open(daily_log, "r") as f:
            for line in f:
                if line.strip():
                    traces.append(json.loads(line))
        
        return traces[-limit:]


# Global instance
_reasoning_engine: Optional[ReasoningEngine] = None

def get_reasoning_engine() -> ReasoningEngine:
    """Get or create the global reasoning engine."""
    global _reasoning_engine
    if _reasoning_engine is None:
        _reasoning_engine = ReasoningEngine()
    return _reasoning_engine