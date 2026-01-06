import json
import re
from typing import Optional

from llm import OllamaLLMIntegration


def _extract_failing_image(error_msg: str) -> Optional[str]:
    if not error_msg:
        return None
    try:
        m = re.search(r'"([^"]+)"', error_msg)
        if m:
            return m.group(1)
        m = re.search(r"manifest for ([^\\s]+) not found", error_msg)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None


def _split_image(image: str) -> tuple[str, str]:
    if "@" in image:
        return image.split("@", 1)[0], ""
    if ":" in image:
        base, tag = image.rsplit(":", 1)
        return base, tag
    return image, ""


def _heuristic_candidates(image: str) -> list[str]:
    base, tag = _split_image(image)
    candidates: list[str] = []
    if tag:
        if tag != "latest":
            candidates.append(f"{base}:latest")
        if tag.endswith("ttt"):
            candidates.append(f"{base}:{tag[:-2]}")
        if "-" in tag:
            candidates.append(f"{base}:{tag.rsplit('-', 1)[0]}")
        if "_" in tag:
            candidates.append(f"{base}:{tag.rsplit('_', 1)[0]}")
        if tag.startswith("v") and len(tag) > 1:
            candidates.append(f"{base}:{tag[1:]}")
    else:
        candidates.append(f"{base}:latest")
    return candidates


def _dedupe(values: list[str], current: str) -> list[str]:
    seen = []
    for val in values:
        if val == current:
            continue
        if val not in seen:
            seen.append(val)
    return seen


def _llm_suggest(image: str, error_msg: str, container_name: str | None) -> tuple[Optional[str], str, float]:
    prompt = (
        "You are analyzing an ImagePullBackOff/ErrImagePull failure in Kubernetes.\n\n"
        f"Image: {image}\n"
        f"Container: {container_name or 'unknown'}\n"
        f"Error: {error_msg}\n\n"
        "Common causes:\n"
        "1. Wrong tag (typo or doesn't exist) -> suggest :latest or previous tag\n"
        "2. Authentication needed -> suggest adding imagePullSecret\n"
        "3. Registry unreachable -> network issue\n"
        "4. Image deleted -> rollback to previous version\n\n"
        "Return JSON with:\n"
        "{\n"
        '  "root_cause": "...",\n'
        '  "recommended_actions": [\n'
        '    {"action": "update_image_url", "image": "nginx:latest", "reason": "..."},\n'
        '    {"action": "add_secret", "secret_name": "...", "reason": "..."},\n'
        '    {"action": "investigate", "reason": "..."}\n'
        "  ]\n"
        "}\n"
        "Only include update_image_url if you are confident the image is correct."
    )
    try:
        llm = OllamaLLMIntegration()
        response = llm.generate_text(prompt=prompt, temperature=0.2, max_tokens=300, json_mode=True)
        response = response.strip()
        if response.startswith("```"):
            response = "\n".join(line for line in response.split("\n") if not line.startswith("```"))
        parsed = json.loads(response) if response else {}
    except Exception:
        return None, "LLM analysis unavailable; using heuristics", 0.4

    recs = parsed.get("recommended_actions") or []
    for rec in recs:
        if rec.get("action") in {"update_image_url", "update_image"}:
            suggested = rec.get("image")
            reason = rec.get("reason") or parsed.get("root_cause") or "LLM recommended image update"
            if suggested and suggested != image:
                return suggested, reason, 0.75
    return None, parsed.get("root_cause") or "LLM did not recommend an image update", 0.45


def analyze_imagepull(image: Optional[str], error_msg: str, container_name: Optional[str]) -> dict:
    """Analyze ImagePullBackOff/ErrImagePull and suggest a safer target image."""
    failing = image or _extract_failing_image(error_msg or "") or ""
    if not failing:
        return {
            "action": "wait_dependency",
            "recommended_image": None,
            "confidence": 0.3,
            "reasoning": "No image reference found in error message",
        }
    if "@" in failing:
        return {
            "action": "wait_dependency",
            "recommended_image": None,
            "confidence": 0.4,
            "reasoning": "Image uses digest; manual intervention required",
        }

    llm_image, llm_reason, llm_conf = _llm_suggest(failing, error_msg or "", container_name)
    if llm_image and llm_image != failing:
        return {
            "action": "fix_image_tag",
            "recommended_image": llm_image,
            "confidence": llm_conf,
            "reasoning": llm_reason,
        }

    candidates = _dedupe(_heuristic_candidates(failing), failing)
    if not candidates:
        return {
            "action": "wait_dependency",
            "recommended_image": None,
            "confidence": 0.4,
            "reasoning": "No safe candidate image found",
        }
    return {
        "action": "fix_image_tag",
        "recommended_image": candidates[0],
        "confidence": 0.6,
        "reasoning": llm_reason if llm_reason else "Heuristic image tag correction",
    }
