import os
import yaml
from functools import lru_cache
from typing import Any, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV = os.getenv("CVA_ENV", "dev").strip().lower() or "dev"


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
        # Fail soft with empty config if parsing fails
        print(f"[config_manager] Failed to load config '{path}': {e}")
        return {}


def get_config(env: str | None = None) -> Dict[str, Any]:
    return load_config(env)


def get_env() -> str:
    return (DEFAULT_ENV or "dev").lower()
