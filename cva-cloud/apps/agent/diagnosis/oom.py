from typing import Optional


def analyze_oom(
    container_name: Optional[str],
    memory_limit: Optional[str],
    exit_code: Optional[int],
) -> dict:
    """Analyze OOMKilled and recommend memory settings."""
    if exit_code == 137:
        return {
            "recommended_memory_request": "256Mi",
            "recommended_memory_limit": "512Mi",
            "reasoning": "Exit code 137 indicates OOMKill",
        }
    if memory_limit:
        return {
            "recommended_memory_request": "256Mi",
            "recommended_memory_limit": memory_limit,
            "reasoning": "Memory limit may be too low",
        }
    return {
        "recommended_memory_request": "256Mi",
        "recommended_memory_limit": "512Mi",
        "reasoning": "OOMKilled detected; increase memory",
    }
