import yaml
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG_PATH = Path(__file__).parent / "agents_config.yaml"


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


def load_agent_config(agent_role: str, mission_type: str | None = None, config_path: Path = DEFAULT_CONFIG_PATH, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
