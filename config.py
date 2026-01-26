"""
CVA Configuration - Centralized settings for timeouts, intervals, and limits.
"""
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class CVAConfig:
    """Central configuration for Catalyst Vector Alpha."""
    
    # === Timing ===
    COGNITIVE_LOOP_INTERVAL: int = 10  # seconds between agent cycles
    IDLE_BACKOFF_BASE: int = 30  # base seconds for idle backoff
    IDLE_BACKOFF_MAX: int = 300  # max seconds for idle backoff
    MISSION_COOLDOWN: int = 30  # seconds between same mission
    MISSION_TIMEOUT: int = 600  # seconds before mission considered stale
    
    # === K8s ===
    K8S_HEALTH_CHECK_INTERVAL: int = 10  # seconds
    K8S_REMEDIATION_COOLDOWN: int = 60  # seconds between remediation attempts
    K8S_REMEDIATION_TTL: int = 600  # seconds to cache remediation state
    
    # === LLM ===
    LLM_TIMEOUT: int = 60  # seconds for LLM calls
    LLM_MAX_TOKENS: int = 1500
    LLM_TEMPERATURE: float = 0.3
    
    # === Memory ===
    MEMORY_DEQUE_MAXLEN: int = 100
    MEMORY_COMPRESSION_THRESHOLD: int = 5  # memories before compression
    
    # === Exploration ===
    EXPLORATION_RATE: float = 0.05  # 5% random exploration
    CURIOSITY_INTERVAL: int = 300  # seconds between curiosity loops
    CURIOSITY_CPU_MAX: float = 60.0  # max CPU usage (%) to allow curiosity
    CURIOSITY_QUIET_MINUTES: int = 10  # minutes since last incident resolution
    
    # === Evolution ===
    EVOLUTION_GAP_THRESHOLD: int = 3  # gaps before evolution
    EVOLUTION_CYCLE_INTERVAL: int = 300  # seconds between evolution checks
    
    # === Agents ===
    AGENT_DEFAULT_COOLDOWN: int = 30  # seconds
    
    # === Paths ===
    PERSISTENCE_DIR: str = "persistence_data"
    LOG_DIR: str = "logs"
    TOOL_CACHE_PATH: str = "persistence_data/tool_embeddings_cache.json"
    
    @classmethod
    def from_env(cls) -> "CVAConfig":
        """Load config with environment variable overrides."""
        return cls(
            COGNITIVE_LOOP_INTERVAL=int(os.getenv("CVA_LOOP_INTERVAL", 10)),
            LLM_TIMEOUT=int(os.getenv("CVA_LLM_TIMEOUT", 60)),
            EXPLORATION_RATE=float(os.getenv("CVA_EXPLORATION_RATE", 0.05)),
            CURIOSITY_INTERVAL=int(os.getenv("CVA_CURIOSITY_INTERVAL", 300)),
            CURIOSITY_CPU_MAX=float(os.getenv("CVA_CURIOSITY_CPU_MAX", 60.0)),
            CURIOSITY_QUIET_MINUTES=int(os.getenv("CVA_CURIOSITY_QUIET_MINUTES", 10)),
            EVOLUTION_GAP_THRESHOLD=int(os.getenv("CVA_EVOLUTION_GAP_THRESHOLD", 3)),
            EVOLUTION_CYCLE_INTERVAL=int(os.getenv("CVA_EVOLUTION_CYCLE_INTERVAL", 300)),
            AGENT_DEFAULT_COOLDOWN=int(os.getenv("CVA_AGENT_DEFAULT_COOLDOWN", 30)),
        )

# Global config instance
config = CVAConfig.from_env()
