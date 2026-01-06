import json
import re
from typing import Optional

from llm import OllamaLLMIntegration


def _extract_env_vars(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text)


def _fallback_analysis(logs: str, exit_code: Optional[int], restarts: int) -> dict:
    logs_lower = logs.lower() if logs else ""
    if exit_code == 137 or "oomkilled" in logs_lower or "exit code 137" in logs_lower:
        return {
            "root_cause": "OOM detected",
            "action": "increase_memory",
            "details": {"memory_request": "256Mi", "memory_limit": "512Mi"},
            "confidence": 0.85,
        }
    if "connection refused" in logs_lower or "cannot connect" in logs_lower:
        return {
            "root_cause": "Dependency not ready",
            "action": "wait_dependency",
            "details": {},
            "confidence": 0.6,
        }
    if "env" in logs_lower or "secret" in logs_lower:
        env_vars = _extract_env_vars(logs)
        return {
            "root_cause": "Missing environment variable",
            "action": "fix_env",
            "details": {"env_vars": env_vars},
            "confidence": 0.65,
        }
    if restarts >= 5:
        return {
            "root_cause": "Repeated crash; rollback may be safer",
            "action": "rollback",
            "details": {},
            "confidence": 0.45,
        }
    return {
        "root_cause": "CrashLoopBackOff without clear signal",
        "action": "wait_dependency",
        "details": {},
        "confidence": 0.4,
    }


def analyze_crashloop(
    logs: str,
    exit_code: Optional[int],
    restarts: int,
    pod_name: str,
) -> dict:
    """Analyze CrashLoopBackOff using LLM (with safe fallback)."""
    prompt = f"""Pod {pod_name} is crashing with CrashLoopBackOff.

Logs (last 50 lines):
{logs[:400] if logs else "No logs available"}

Exit code: {exit_code}
Restart count: {restarts}

Common patterns:
- Exit 137 = SIGKILL (hidden OOM)
- "connection refused" = dependency not ready
- "permission denied" = security context issue
- Missing env vars cause startup crash

Analyze the root cause and recommend ONE action.

Respond ONLY with valid JSON in this exact format:
{{
  "root_cause": "brief explanation",
  "action": "increase_memory|rollback|fix_env|wait_dependency",
  "details": {{
    "memory_request": "256Mi",
    "memory_limit": "512Mi",
    "env_vars": ["VAR_NAME"],
    "reason": "why this fix"
  }}
}}
"""
    try:
        llm = OllamaLLMIntegration()
        response = llm.generate_text(prompt=prompt, temperature=0.2, max_tokens=500, json_mode=True)
        response = response.strip()
        if response.startswith("```"):
            response = "\n".join(line for line in response.split("\n") if not line.startswith("```"))
        analysis = json.loads(response) if response else {}
    except Exception:
        return _fallback_analysis(logs, exit_code, restarts)

    action = analysis.get("action") or "wait_dependency"
    if action not in {"increase_memory", "rollback", "fix_env", "wait_dependency"}:
        return _fallback_analysis(logs, exit_code, restarts)

    details = analysis.get("details") or {}
    if action == "fix_env" and not details.get("env_vars"):
        details["env_vars"] = _extract_env_vars(logs)
    if action == "increase_memory":
        details.setdefault("memory_request", "256Mi")
        details.setdefault("memory_limit", "512Mi")

    return {
        "root_cause": analysis.get("root_cause") or details.get("reason") or "CrashLoopBackOff analysis",
        "action": action,
        "details": details,
        "confidence": 0.8,
    }
