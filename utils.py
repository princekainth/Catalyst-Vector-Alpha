# utils.py
from __future__ import annotations

"""
General utility functions for the entire system (production grade).
- Plan building prompt (JSON-only, tool-specific guidance)
- Plan normalization with strict policy validation
- Step dispatch with consistent breadcrumbs
- Hardened JSON extraction/repair helpers
- Safe timeouts (POSIX + graceful Windows fallback)
"""

import json
import logging
import re
import signal
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Optional, Iterable

import requests
import hashlib


from core.policy import (
    validate_task_intent,             # back-compat (title-only)
    validate_step_intent,            # preferred (step-aware)
    validate_role_tool_assignment,   # back-compat (role ↔ tool)
    validate_role_task_assignment,   # role ↔ task_type
    resolve_task_type,               # tool/step → task_type
    explain_step_policy,             # structured diagnostics
    infer_agent_role,
)

LOGGER = logging.getLogger(__name__)


# very light placeholder set so this module doesn’t depend on core.policy internals
_PLACEHOLDER_TOKENS = {"", " ", "tbd", "todo", "none", "null", "n/a", "na", "unspecified", "no specific intent"}

def _title_of(step: dict) -> Optional[str]:
    """
    Prefer 'title'; fall back to 'description'. Strip and reject placeholders.
    """
    if not isinstance(step, dict):
        return None
    raw = step.get("title") or step.get("description") or ""
    title = str(raw).strip()
    if not title or title.lower() in _PLACEHOLDER_TOKENS:
        return None
    return title

def _tools_list_candidates(step: dict, available_tools: Iterable[str]) -> list[str]:
    """
    Collect *valid* tool candidates appearing in either 'tool' or 'tools'.
    Deduped, order-preserving where possible.
    """
    if not isinstance(step, dict):
        return []
    avail = set(t.strip() for t in available_tools if isinstance(t, str))
    out: list[str] = []

    # single
    t = step.get("tool")
    if isinstance(t, str):
        ts = t.strip()
        if ts and ts in avail:
            out.append(ts)

    # list
    tl = step.get("tools")
    if isinstance(tl, list):
        for x in tl:
            if not isinstance(x, str):
                continue
            xs = x.strip()
            if xs and xs in avail and xs not in out:
                out.append(xs)

    return out

def _pick_tool(step: dict, available_tools: Iterable[str]) -> Optional[str]:
    """
    Concrete selection logic:
      - if step['tool'] is valid → use it
      - else if a single valid candidate exists in step['tools'] → use it
      - else → None (either invalid or ambiguous)
    """
    if not isinstance(step, dict):
        return None
    avail = set(t.strip() for t in available_tools if isinstance(t, str))

    # explicit single
    t = step.get("tool")
    if isinstance(t, str):
        ts = t.strip()
        if ts in avail:
            return ts

    # candidates in list
    cands = _tools_list_candidates(step, avail)
    if len(cands) == 1:
        return cands[0]

    return None  # none or ambiguous

def _norm_args_for_tool(tool: str, args: Any) -> Optional[dict]:
    """
    Minimal normalization:
      - require dict (else None to signal unrecoverable)
      - drop keys with None values
      - apply a couple of safe defaults for known tools (optional but handy)
    """
    if not isinstance(args, dict):
        return None
    clean = {k: v for k, v in args.items() if v is not None}

    # Safe, optional convenience defaults:
    if tool == "get_system_cpu_load":
        # default 300s window if not provided
        clean.setdefault("time_interval_seconds", 1)

    return clean

def _step_key(agent: str, tool: str, title: str, args: Dict[str, Any]) -> str:
    """
    Stable dedup key for a step.
    Use sorted JSON of args to avoid ordering differences.
    """
    payload = {
        "agent": agent,
        "tool": tool,
        "title": title,
        "args": args,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def _log_policy_skip(step_index: int, original_step: dict, expl: dict) -> None:
    """
    Emit a single structured line for observability.
    """
    try:
        log_obj = {
            "step_index": step_index,
            "title": original_step.get("title") or original_step.get("description"),
            "agent": original_step.get("agent"),
            "tool": original_step.get("tool") or (original_step.get("tools") if isinstance(original_step.get("tools"), list) else None),
            "intent_ok": expl.get("intent_ok"),
            "tool_ok": expl.get("tool_ok"),
            "role_ok": expl.get("role_ok"),
            "intent": expl.get("intent"),
            "task_type": expl.get("task_type"),
            "reasons": expl.get("reasons", []),
        }
        LOGGER.info("PLAN_STEP_SKIPPED %s", json.dumps(log_obj, ensure_ascii=False))
    except Exception as e:
        # never let logging break flow
        LOGGER.warning("PLAN_STEP_SKIPPED (unstructured): index=%s error=%r expl=%r", step_index, e, expl)


# ------------------------------
# Schema hints for planners
# ------------------------------

PLAN_SCHEMA_EXAMPLE = {
    "summary": "A brief, one-sentence description of the overall plan.",
    "steps": [
        {
            "id": "S1",
            "title": "A short description of this specific step.",
            "agent": "The single most appropriate agent for this step.",
            "tool": "A specific tool_name from the provided list, or null if no tool is needed.",
            "args": {"key": "value"},
            "depends_on": []
        }
    ]
}

def _json_schema_to_text() -> str:
    return json.dumps(PLAN_SCHEMA_EXAMPLE, indent=2)


def build_plan_prompt(goal_str: str, agents: list[str], tool_instructions: str) -> str:
    """
    Toolsmith-enabled planner prompt - can write code when needed.
    """
    schema_text = _json_schema_to_text()
    agents_csv = ", ".join(sorted(agents))
    
    # Check if goal requires building new capability
    needs_toolsmith = any(keyword in goal_str.lower() for keyword in [
        'memory usage', 'disk space', 'system check', 'internet speed', 
        'scrape', 'parse', 'analyze file', 'process data'
    ])
    
    if needs_toolsmith:
        mode_instruction = """
🔧 TOOLSMITH MODE ACTIVATED

This goal requires capabilities not in your standard toolset.
BUILD the solution using execute_terminal_command:

Example plan structure:
{
  "summary": "Check memory usage via custom script",
  "steps": [
    {
      "id": "S1",
      "title": "Write memory check script",
      "agent": "ProtoAgent_Worker_instance_1",
      "tool": "execute_terminal_command",
      "args": {"command": "cat > /workspace/mem.py << 'EOF'\\nimport psutil\\nprint(f'Memory: {psutil.virtual_memory().percent}%')\\nEOF"},
      "depends_on": []
    },
    {
      "id": "S2", 
      "title": "Run memory check script",
      "agent": "ProtoAgent_Worker_instance_1",
      "tool": "execute_terminal_command",
      "args": {"command": "python3 /workspace/mem.py"},
      "depends_on": ["S1"]
    }
  ]
}
"""
    else:
        mode_instruction = "Use existing tools from the list below to accomplish the goal."
    
    return "\n".join([
        "You are an autonomous planner. Respond with JSON ONLY.",
        "",
        f'GOAL: "{goal_str}"',
        "",
        mode_instruction,
        "",
        f"AVAILABLE_AGENTS: {agents_csv}",
        "",
        "KEY TOOLS:",
        "- execute_terminal_command: Run any shell command in sandbox",
        "- check_calendar: Check Google Calendar events",
        "- web_search: Search the web",
        "- k8s_scale: Scale Kubernetes deployments",
        "",
        "SCHEMA:",
        schema_text,
        "",
        "RULES:",
        "- Each step must use exactly ONE tool",
        "- Provide meaningful argument values",
        "- Steps can depend on previous steps",
        "- Maximum 12 steps",
    ])

# ------------------------------
# Safe timeout helper
# ------------------------------

@contextmanager
def timeout(seconds: int):
    """
    POSIX timeout using SIGALRM; gracefully no-op on platforms without SIGALRM (e.g., Windows).
    """
    has_sigalrm = hasattr(signal, "SIGALRM")
    if not has_sigalrm or seconds <= 0:
        # Best-effort: no hard kill, but still provides a uniform context manager API.
        yield
        return

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    prev_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


# ------------------------------
# Plan validation & normalization
# ------------------------------

def _title_of(s: dict) -> str:
    return (s.get("title") or s.get("description") or "").strip()


def _pick_tool(s: dict, available_tools: set[str]) -> Optional[str]:
    """
    Choose the single concrete tool:
    - Prefer `tool` if present and valid
    - Else scan `tools` list for the first valid candidate
    """
    t = (s.get("tool") or "").strip()
    if t:
        return t if t in available_tools else None

    if isinstance(s.get("tools"), list):
        for cand in s["tools"]:
            cand = (cand or "").strip()
            if cand in available_tools:
                return cand

    return None


def _tools_list_candidates(s: dict, available_tools: set[str]) -> List[str]:
    """
    Sanitize & filter the provided `tools` candidates to valid ones.
    """
    tools_raw = s.get("tools") or []
    if not isinstance(tools_raw, list):
        return []
    cands = []
    for t in tools_raw:
        if not isinstance(t, str):
            continue
        tt = t.strip()
        if tt in available_tools:
            cands.append(tt)
    return cands


def _norm_args_for_tool(tool: str, args: Any) -> Optional[dict]:
    """
    Tool-specific argument normalization (extend as needed).
    Returns a dict or None if unrecoverable.
    """
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None

    # Example: create_pdf normalizations
    if tool == "create_pdf":
        if "content" in args and "text_content" not in args:
            args["text_content"] = args.pop("content")
        if "file_name" in args and "filename" not in args:
            args["filename"] = args.pop("file_name")

    return args


def _step_key(agent: str, tool: str, title: str, args: dict) -> str:
    return json.dumps(
        {"agent": agent, "tool": tool, "title": title, "args": args},
        sort_keys=True,
        ensure_ascii=False,
    )

def _normalize_plan_schema(
    self,
    plan: dict,
    available_agents: set[str],
    available_tools: set[str],
    max_steps: int = 12,
) -> tuple[list[dict], dict]:
    """
    Accepts either:
      - new schema: {title, agent, tool, args}
      - legacy schema: {description, tools: [..]}
    Returns (clean_steps, skips) with exactly one valid tool per step.
    """
    steps = plan.get("steps") or []
    clean: list[dict] = []
    skips = {
        "bad_agent": 0,
        "bad_tool": 0,
        "bad_args": 0,
        "empty": 0,
        "multi_tool": 0,
        "duplicate": 0,
        "bad_intent": 0,
        "role_mismatch": 0,
    }
    seen: set[str] = set()

    # Precompute inferred roles for available agents (used for reassignment)
    inferred_roles: dict[str, str] = {a: infer_agent_role(a) for a in available_agents}

    # --- role helper pickers (scoped to this function) ----------------------
    def _is_worker(a: str) -> bool:
        return (inferred_roles.get(a) or infer_agent_role(a)) == "tool_using_executor"

    def _pick_worker(agents: set[str]) -> str | None:
        for name in ("ProtoAgent_Worker_instance_1", "ProtoAgent_Worker"):
            if name in agents:
                return name
        for a in agents:
            if _is_worker(a):
                return a
        return None

    def _is_observer(a: str) -> bool:
        role = (inferred_roles.get(a) or infer_agent_role(a))
        return role in ("observer", "sensor")

    def _pick_observer(agents: set[str]) -> str | None:
        for name in ("ProtoAgent_Observer_instance_1", "ProtoAgent_Observer"):
            if name in agents:
                return name
        for a in agents:
            if _is_observer(a):
                return a
        return None
    # -----------------------------------------------------------------------

    for i, s in enumerate(steps[:max_steps], start=1):
        if not isinstance(s, dict):
            skips["empty"] += 1
            continue

        title = _title_of(s)
        if not title:
            skips["empty"] += 1
            continue

        # --- Belt & suspenders intent/mission stamping -----------------------
        intent_val = (
            s.get("intent")
            or s.get("strategic_intent")
            or s.get("mission_type")
            or plan.get("mission_type")
            or plan.get("strategic_intent")
            or "health_audit"
        )
        s.setdefault("intent", intent_val)
        s.setdefault("mission_type", plan.get("mission_type") or intent_val)
        s.setdefault("strategic_intent", s["mission_type"])
        # ---------------------------------------------------------------------

        # If planner emitted multiple tools but no single chosen one, catch early.
        tools_valid = _tools_list_candidates(s, available_tools)
        if not s.get("tool") and len(tools_valid) > 1:
            skips["multi_tool"] += 1
            continue

        agent = (s.get("agent") or "ProtoAgent_Worker_instance_1").strip()
        if agent not in available_agents:
            skips["bad_agent"] += 1
            _log_policy_skip(i, s, {
                "intent_ok": None, "agent_ok": False, "tool_ok": None, "role_ok": None,
                "reasons": [f"Unknown or missing agent '{agent}'."]
            })
            continue

        tool = _pick_tool(s, available_tools)
        if not tool:
            # differentiate between multi-tool vs invalid tool for better observability
            if isinstance(s.get("tools"), list) and len(tools_valid) > 1:
                skips["multi_tool"] += 1
            else:
                skips["bad_tool"] += 1
            _log_policy_skip(i, s, {
                "intent_ok": None, "agent_ok": True, "tool_ok": False, "role_ok": None,
                "reasons": ["No concrete usable tool selected or tool not allowed."]
            })
            continue

        # Normalize args (tool-specific fixups)
        args = _norm_args_for_tool(tool, s.get("args", {}))
        if args is None:
            skips["bad_args"] += 1
            _log_policy_skip(i, s, {
                "intent_ok": None, "agent_ok": True, "tool_ok": True, "role_ok": None,
                "reasons": ["Args must be a dict; unrecoverable for this tool."]
            })
            continue

        # --- role hardmaps BEFORE policy explain --------------------------------
        # Ensure certain tools land on the right role (best-effort) before validation.
        if tool in {"update_resource_allocation"} and not _is_worker(agent):
            repl = _pick_worker(available_agents)
            if repl and repl != agent:
                LOGGER.info(
                    "PLAN_NORMALIZATION_REASSIGN step=%s title=%r from_agent=%r to_agent=%r reason=%s",
                    i, title, agent, repl, f"Tool '{tool}' requires worker role"
                )
                agent = repl

        if tool in {"measure_responsiveness", "get_system_cpu_load", "top_processes"} and not _is_observer(agent):
            repl = _pick_observer(available_agents)
            if repl and repl != agent:
                LOGGER.info(
                    "PLAN_NORMALIZATION_REASSIGN step=%s title=%r from_agent=%r to_agent=%r reason=%s",
                    i, title, agent, repl, f"Tool '{tool}' prefers observer role"
                )
                agent = repl
        # -------------------------------------------------------------------------

        # Use policy explainer for robust validation (intent/tool/role)
        expl = explain_step_policy({
            "title": title,
            "agent": agent,
            "tool": tool,
            "args": args,
            "intent": intent_val,                                # inferred value
            "task_type": s.get("task_type"),
            "mission_type": s.get("mission_type") or intent_val, # stable mission
        })

        # If role fails but intent & tool are fine, try auto-reassignment to a compliant agent.
        if expl.get("intent_ok") and expl.get("tool_ok") and not expl.get("role_ok"):
            required_task_type = expl.get("task_type")
            replacement_agent = None
            for candidate in available_agents:
                role = inferred_roles.get(candidate) or infer_agent_role(candidate)
                if validate_role_task_assignment(role, required_task_type):
                    replacement_agent = candidate
                    break
            if replacement_agent and replacement_agent != agent:
                LOGGER.info(
                    "PLAN_NORMALIZATION_REASSIGN step=%s title=%r from_agent=%r to_agent=%r reason=%s",
                    i, title, agent, replacement_agent,
                    f"Role mismatch for task_type '{required_task_type}'"
                )
                agent = replacement_agent
                # Re-check policy after reassignment
                expl = explain_step_policy({
                    "title": title,
                    "agent": agent,
                    "tool": tool,
                    "args": args,
                    "intent": intent_val,
                    "task_type": s.get("task_type"),
                    "mission_type": s.get("mission_type") or intent_val,
                })

        # Map expl flags → skip counters
        failed = False
        if not expl.get("intent_ok", False):
            skips["bad_intent"] += 1
            failed = True
        if not expl.get("tool_ok", False):
            skips["bad_tool"] += 1
            failed = True
        if not expl.get("role_ok", False):
            skips["role_mismatch"] += 1
            failed = True

        if failed:
            _log_policy_skip(i, s, expl)
            continue

        # Duplicate elimination
        key = _step_key(agent, tool, title, args)
        if key in seen:
            skips["duplicate"] += 1
            continue
        seen.add(key)

        clean.append({"title": title, "agent": agent, "tool": tool, "args": args})

    return clean, skips


async def _dispatch_plan_steps_async(self, plan: dict, goal_str: str) -> int:
    """Inject directives and optionally execute them in parallel."""
    plan_id = plan.get("id") or f"plan-{uuid.uuid4()}"
    steps = plan.get("steps") or []
    directives = []

    for i, s in enumerate(steps, start=1):
        agent_field = s.get("agent") or s.get("agent_name")
        tool_field  = s.get("tool")  or s.get("tool_name")
        args_field  = s.get("args")  or s.get("tool_args")

        directive = {
            "id": f"dir-{uuid.uuid4()}",
            "type": "AGENT_PERFORM_TASK",
            "agent_name": agent_field,
            "task_description": s["title"],
            "tool_name": tool_field,
            "tool_args": args_field or {},
            "context": {
                "plan_id": plan_id,
                "parent_goal": goal_str,
                "step": i,
                "total_steps": len(steps),
            },
        }

        self._log_agent_activity(
            "PLAN_STEP_INJECT",
            self.name,
            {
                "plan_id": plan_id,
                "step": i,
                "agent": directive["agent_name"],
                "tool": directive["tool_name"],
                "title": directive["task_description"],
            },
        )
        directives.append(directive)

    # Inject all directives
    if directives:
        self.orchestrator.inject_directives(directives)

    # NEW: Execute independent steps in parallel
    import asyncio
    independent_steps = [d for d in directives if _is_independent_step(self, d)]
    
    if independent_steps and len(independent_steps) > 1:
        print(f"  [ASYNC] Executing {len(independent_steps)} steps in parallel")
        tasks = [_execute_directive_async(self, d) for d in independent_steps]
        await asyncio.gather(*tasks, return_exceptions=True)

    self._log_agent_activity(
        "PLAN_STEPS_INJECTED",
        self.name,
        {"plan_id": plan_id, "count": len(directives)}
    )

    return len(directives)

def _dispatch_plan_steps(self, plan: dict, goal_str: str) -> int:
    """Sync wrapper for async dispatch."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_dispatch_plan_steps_async(self, plan, goal_str))
    except RuntimeError:
        return asyncio.run(_dispatch_plan_steps_async(self, plan, goal_str))

def _is_independent_step(self, directive: dict) -> bool:
    """Check if step can run in parallel (no dependencies)."""
    # Simple heuristic: read-only tools can run in parallel
    tool = directive.get("tool_name", "")
    parallel_safe = ["web_search", "get_system_cpu_load", "kubernetes_pod_metrics", 
                     "read_webpage", "measure_responsiveness"]
    return tool in parallel_safe

async def _execute_directive_async(self, directive: dict):
    """Execute a single directive asynchronously."""
    # This would integrate with your directive execution system
    pass


# ------------------------------
# JSON extraction/repair
# ------------------------------

def extract_json_candidates(text: str) -> List[str]:
    """
    Extract potential JSON using multiple strategies:
    1) ```json ... ``` fenced blocks
    2) ``` ... ``` generic fenced blocks
    3) First balanced {...} block
    4) Whole text if it looks like JSON
    """
    candidates: List[str] = []

    # 1) ```json fenced blocks
    for m in re.finditer(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(m.group(1).strip())

    # 2) generic ``` fenced blocks (no language tag)
    for m in re.finditer(r"```\s*(.*?)\s*```", text, flags=re.DOTALL):
        block = m.group(1).strip()
        if block.startswith("{") and block.endswith("}"):
            candidates.append(block)

    # 3) first balanced {...}
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i+1].strip())
                    break

    # 4) whole text if JSON-ish
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        candidates.append(t)

    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def sanitize_json_string(s: str) -> str:
    """Light cleaning on a potential JSON string before parsing."""
    s = s.lstrip('\ufeff')  # BOM
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)  # control chars
    s = re.sub(r',\s*([}\]])', r'\1', s)  # trailing commas
    return s.strip()


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON using multiple extraction and sanitization strategies."""
    for candidate in extract_json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                sanitized_candidate = sanitize_json_string(candidate)
                return json.loads(sanitized_candidate)
            except json.JSONDecodeError:
                continue
    return None


def extract_json_from_text(text: str) -> str:
    """Greedy slice from the first '{' to the last '}' (fallback helper)."""
    start = text.find('{')
    end = text.rfind('}') + 1
    if start != -1 and end != 0 and start < end:
        return text[start:end]
    return ""


def validate_json_structure(data: dict, required_keys: list) -> bool:
    return all(key in data for key in required_keys)


def safe_json_loads(json_string: str, default: Any = None) -> Any:
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default


def ollama_chat(model: str,
                messages: List[Dict[str, str]],
                format_json: bool = False,
                temperature: float = 0.2,
                timeout_seconds: int = 120) -> str:
    """
    Thread-safe Ollama chat client (no SIGALRM).
    Uses requests' per-call timeout and handles common Ollama response shapes.
    """
    from requests.exceptions import RequestException, Timeout

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": False,
    }
    if format_json:
        payload["format"] = "json"

    try:
        resp = requests.post(url, json=payload, timeout=timeout_seconds)
        resp.raise_for_status()

        content_type = (resp.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            data = resp.json()

            m = data.get("message")
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                return m["content"]

            r = data.get("response")
            if isinstance(r, str):
                return r

            msgs = data.get("messages")
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                if isinstance(last, dict) and isinstance(last.get("content"), str):
                    return last["content"]

            return resp.text

        return resp.text

    except Timeout:
        raise TimeoutError(f"Ollama request timed out after {timeout_seconds} seconds")
    except RequestException as e:
        body = ""
        try:
            if e.response is not None:
                body = f" | body: {e.response.text[:300]}"
        except Exception:
            pass
        raise ConnectionError(f"Ollama connection failed: {e}{body}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during Ollama call: {str(e)}")


def llm_fix_json_response(raw_text: str, model: str = "mistral-small") -> Optional[Dict[str, Any]]:
    """
    Use LLM to repair malformed JSON responses (last resort).
    """
    schema_text = _json_schema_to_text()
    repair_prompt = f"""
Convert the following content into STRICT, valid JSON that matches this shape EXACTLY:

{schema_text}

Rules:
- Return ONLY the JSON object (no prose, no fences).
- Fix trailing commas, quotes, and escaping.
- Preserve keys and required fields per the shape.

CONTENT:
<<<
{raw_text[:2000]}
>>>
""".strip()

    try:
        repaired_text = ollama_chat(
            model=model,
            messages=[{"role": "user", "content": repair_prompt}],
            format_json=True,
            temperature=0.0,
            timeout_seconds=120
        )
        return try_parse_json(repaired_text)
    except Exception as e:
        LOGGER.warning("JSON repair via LLM failed: %s", str(e))
        return None


def safe_truncate(text: str, max_length: int = 500, suffix: str = "...[truncated]") -> str:
    if len(text) <= max_length:
        return text
    truncated = text[:max_length - len(suffix)]
    while truncated and not truncated[-1].isprintable():
        truncated = truncated[:-1]
    return truncated + suffix

def validate_plan_shape(plan: dict, available_agents: set, available_tools: set) -> tuple[bool, str]:
    """
    Lenient validation for pre-normalization plans.
    This first-pass check validates the basic structure and types of a plan
    before it undergoes full normalization, which resolves concrete agent/tool
    assignments.
    """
    if not isinstance(plan, dict):
        return False, "Plan must be a dict."

    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return False, "Plan.steps must be a non-empty list."

    for i, s in enumerate(steps, start=1):
        if not isinstance(s, dict):
            return False, f"Step {i} must be a dict."

        # Title/description presence (required)
        title = _title_of(s)
        if not title:
            return False, f"Step {i} missing 'title' or 'description'."

        # Agent: allow missing/unknown here (normalizer will default/validate)
        agent = (s.get("agent") or "").strip()
        if "agent" in s and not isinstance(s["agent"], str):
            return False, f"Step {i} has non-string agent."

        # Tool: allow None/unknown here; just ensure types are sane
        tool_val = s.get("tool", None)
        if "tool" in s and tool_val is not None and not isinstance(tool_val, str):
            return False, f"Step {i} has non-string tool."

        tools_list = s.get("tools", None)
        if "tools" in s and tools_list is not None and not isinstance(tools_list, list):
            return False, f"Step {i} has non-list tools."

        # Args must be a dict if present
        if "args" in s and not isinstance(s["args"], dict):
            return False, f"Step {i} has non-dict args."

        # Optional fields: type checks only
        if "id" in s and not isinstance(s["id"], str):
            return False, f"Step {i} has non-string id."
        if "depends_on" in s and not isinstance(s["depends_on"], list):
            return False, f"Step {i} has non-list depends_on."

    return True, "ok"


def plan_has_actionable_step(plan: Dict[str, Any], available_tools: set) -> Tuple[bool, str]:
    """
    Ensure at least one step is executable: has a known tool or explicit args.
    """
    steps = plan.get("steps", [])
    for s in steps:
        tool = (s.get("tool") or "").strip()
        if tool in available_tools:
            return True, "Found step with concrete tool."
        if not tool and isinstance(s.get("args"), dict) and s["args"]:
            return True, "Found step with explicit args."
    return False, "No actionable steps (all tools null or unknown and no args)."


def normalize_then_dispatch(self, raw_plan: dict, goal_str: str,
                            available_agents: set[str], available_tools: set[str]) -> int:
    clean, skips = self._normalize_plan_schema(raw_plan, available_agents, available_tools)
    # Compact summary for logs/metrics
    LOGGER.info(
        "PLAN_NORMALIZATION_SUMMARY bad_agent=%s bad_tool=%s bad_args=%s empty=%s multi_tool=%s duplicate=%s bad_intent=%s role_mismatch=%s",
        skips.get("bad_agent"), skips.get("bad_tool"), skips.get("bad_args"),
        skips.get("empty"), skips.get("multi_tool"), skips.get("duplicate"),
        skips.get("bad_intent"), skips.get("role_mismatch"),
    )
    plan = {"id": raw_plan.get("id"), "steps": clean}
    return self._dispatch_plan_steps(plan, goal_str)
