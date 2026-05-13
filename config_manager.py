"""
Unified CVA Configuration Manager.

Consolidates config.py, config_loader.py, and config_manager.py into a single
module. All three original files now re-export from here for backward compat.
"""
import os
import yaml
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("ConfigManager")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV = os.getenv("CVA_ENV", "dev").strip().lower() or "dev"

# ============================================================
# YAML-based env config (was config_manager.py)
# ============================================================

@lru_cache(maxsize=4)
def load_config(env: str | None = None) -> Dict[str, Any]:
    """Load YAML config for the requested environment (default: CVA_ENV or dev)."""
    target_env = (env or DEFAULT_ENV).lower()
    config_dir = os.path.join(BASE_DIR, "config")
    path = os.path.join(config_dir, f"{target_env}.yaml")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning("Failed to load config '%s': %s", path, e)
        return {}


def get_config(env: str | None = None) -> Dict[str, Any]:
    return load_config(env)


def get_env() -> str:
    return (DEFAULT_ENV or "dev").lower()


# ============================================================
# Dataclass config (was config.py)
# ============================================================

@dataclass
class CVAConfig:
    """Central configuration for Catalyst Vector Alpha."""

    # === Timing ===
    COGNITIVE_LOOP_INTERVAL: int = 10
    IDLE_BACKOFF_BASE: int = 30
    IDLE_BACKOFF_MAX: int = 300
    MISSION_COOLDOWN: int = 30
    MISSION_TIMEOUT: int = 600

    # === K8s ===
    K8S_HEALTH_CHECK_INTERVAL: int = 10
    K8S_REMEDIATION_COOLDOWN: int = 60
    K8S_REMEDIATION_TTL: int = 600

    # === LLM ===
    LLM_TIMEOUT: int = 60
    LLM_MAX_TOKENS: int = 1500
    LLM_TEMPERATURE: float = 0.3

    # === Memory ===
    MEMORY_DEQUE_MAXLEN: int = 100
    MEMORY_COMPRESSION_THRESHOLD: int = 5

    # === Exploration ===
    EXPLORATION_RATE: float = 0.05
    CURIOSITY_INTERVAL: int = 300
    CURIOSITY_CPU_MAX: float = 60.0
    CURIOSITY_QUIET_MINUTES: int = 10

    # Evolution Agent
    EVOLUTION_GAP_THRESHOLD: int = 1
    EVOLUTION_CYCLE_INTERVAL: int = 60

    # Self-Healing
    SELF_HEALING_CHECK_INTERVAL: int = 600

    # Students / K8s
    K8S_STUDENT_COOLDOWN: int = 300
    K8S_STUDENT_MAX_ATTEMPTS: int = 3

    # Guardian
    GUARDIAN_MAX_AGENTS: int = 50
    GUARDIAN_MAX_FAILED_TASKS: int = 3
    GUARDIAN_MAX_IDLE_MINS: int = 30

    # === Agents ===
    AGENT_DEFAULT_COOLDOWN: int = 30
    DEMO_MODE: bool = False

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
            SELF_HEALING_CHECK_INTERVAL=int(os.getenv("CVA_SELF_HEALING_INTERVAL", 600)),
            K8S_STUDENT_COOLDOWN=int(os.getenv("CVA_K8S_STUDENT_COOLDOWN", 300)),
            K8S_STUDENT_MAX_ATTEMPTS=int(os.getenv("CVA_K8S_STUDENT_MAX_ATTEMPTS", 3)),
            GUARDIAN_MAX_AGENTS=int(os.getenv("CVA_GUARDIAN_MAX_AGENTS", 50)),
            GUARDIAN_MAX_FAILED_TASKS=int(os.getenv("CVA_GUARDIAN_MAX_FAILED_TASKS", 3)),
            GUARDIAN_MAX_IDLE_MINS=int(os.getenv("CVA_GUARDIAN_IDLE_MINS", 30)),
            DEMO_MODE=os.getenv("CVA_DEMO_MODE") == "1",
        )


# Global config instance (backward compat for `from config import config`)
config = CVAConfig.from_env()


def get_cva_config() -> CVAConfig:
    """Return the global CVAConfig dataclass instance."""
    return config


# ============================================================
# Agent config loader (was config_loader.py)
# ============================================================

DEFAULT_AGENT_CONFIG_PATH = Path(__file__).parent / "agents_config.yaml"


def _llm_infer_defaults(agent_role: str, mission_type: str | None, known: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to infer sensible defaults for missing values based on role + mission."""
    try:
        from shared_models import OllamaLLMIntegration
        llm = OllamaLLMIntegration()
        prompt = f"""You are CVA's config default reasoner.
Agent role: {agent_role}
Mission type: {mission_type or 'unknown'}
Known config: {json.dumps(known, ensure_ascii=False)}

Propose a minimal JSON object with sensible defaults for missing fields. Only include keys you infer; leave known keys untouched.
Keep values simple (strings, numbers, booleans)."""
        resp = llm.generate_text(prompt=prompt, json_mode=True, temperature=0.2, max_tokens=200)
        inferred = json.loads(resp) if resp else {}
        return inferred if isinstance(inferred, dict) else {}
    except Exception:
        return {}


def load_agent_config(
    agent_role: str,
    mission_type: str | None = None,
    config_path: Path = DEFAULT_AGENT_CONFIG_PATH,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Load default config for the given agent role from agents_config.yaml.
    Missing keys may be inferred via LLM. Overrides can be supplied to prefill values.
    """
    try:
        if not config_path.exists():
            return {}
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        role_key = (agent_role or "").strip().lower()
        cfg = data.get(role_key) or {}
        if not isinstance(cfg, dict):
            cfg = {}
        # Merge in shared block
        shared = data.get("shared") or {}
        if isinstance(shared, dict):
            merged = dict(shared)
            merged.update(cfg)
            cfg = merged
        # Apply overrides
        if overrides:
            cfg.update(overrides)
        # LLM inference for missing
        inferred = _llm_infer_defaults(role_key, mission_type, cfg)
        for k, v in (inferred or {}).items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    except Exception:
        return {}
