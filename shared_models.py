# ==================================================================
#  shared_models.py - Core Components for Catalyst Vector Alpha
# ==================================================================
from __future__ import annotations

"""Core models and utilities used across Catalyst Vector Alpha."""

# --- Standard Library Imports ---
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterable, Union, Deque, Tuple, TYPE_CHECKING, TypeAlias
import logging
import json
import os
import uuid
import random
import collections
import collections.abc
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import re
import copy
from threading import Lock

# --- Project Imports ---
from tool_registry import ToolRegistry
from tools import (
    get_system_cpu_load_tool,
    initiate_network_scan_tool,
    deploy_recovery_protocol_tool,
    update_resource_allocation_tool,
    get_environmental_data_tool,
)

# --- Third-Party Library Imports ---
# Guard ChatResponse for environments where ollama._types may not exist.
try:  # Prefer real type when available
    from ollama._types import ChatResponse as _OllamaChatResponse  # type: ignore
    ChatResponse: TypeAlias = _OllamaChatResponse
except Exception:  # Fallback keeps type checkers happy
    ChatResponse: TypeAlias = Dict[str, Any]

import yaml
import ollama
import chromadb
import jsonschema
import psutil


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def utc_now_iso() -> str:
    """UTC timestamp in RFC3339-ish format with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def get_logger(name: str = "CatalystLogger") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # remove old handlers once
    for h in list(logger.handlers):
        logger.removeHandler(h)

    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(h)
    return logger


# Module-level logger (use everywhere in this file)
logger = get_logger("CatalystLogger")

# --- Globals used across the module (optional) ---
system_instance: Optional[Any] = None
main_app_logger: Optional[logging.Logger] = None


# ==================================================================
#  1. Communication & Event Handling
# ==================================================================

from messaging import BusMessage, MessageBus, EventMonitor  # noqa: E402
# ==================================================================
#  2. Agent Memory & Cognition (extracted to memetic_kernel.py)
# ==================================================================
from memetic_kernel import MemeticKernel, timestamp_now  # noqa: E402


# ==================================================================
#  3. Tools and Tool Registry
# ==================================================================

class Tool:
    """
    Represents an external function or API call that an agent can use.
    The schema adheres to common LLM function calling conventions.
    """
    def __init__(self, name: str, description: str, parameters: dict, func):
        self.name = name
        self.description = description
        self.parameters = parameters # JSON schema-like dictionary for parameters
        self.func = func # The actual Python function to call

    def get_function_spec(self) -> dict:
        """Returns the tool's specification in a format suitable for LLM context."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def __call__(self, *args, **kwargs):
        """Allows the Tool instance to be called directly, executing its wrapped function."""
        return self.func(*args, **kwargs)

# ==================================================================
#  4. System Configuration & LLM Integration
# ==================================================================
class ISLSchemaValidator:
    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        try:
            with open(schema_path, "r") as f:
                self.schema = yaml.safe_load(f)
            if not isinstance(self.schema, dict) or "directives" not in self.schema:
                raise ValueError("ISL schema must be a dict with a top-level 'directives' key.")
            # Precompile validator for speed; Draft7 to match your current usage
            self.validator = jsonschema.Draft7Validator(self.schema)
            logger.info(f"Loaded ISL schema from {schema_path}")
        except FileNotFoundError:
            raise ValueError(f"ISL Schema file not found at: {schema_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing ISL Schema YAML: {e}")
        except jsonschema.exceptions.SchemaError as e:
            raise ValueError(f"Invalid ISL Schema itself: {e.message} at {list(e.path)}")

    def validate_manifest(self, manifest: dict) -> bool:
        """
        Validates a manifest against the loaded ISL schema using jsonschema.
        Raises ValueError with detailed error messages if validation fails.
        """
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be an object.")
        if "directives" not in manifest:
            raise ValueError("Manifest must contain a top-level 'directives' key.")
        if not isinstance(manifest["directives"], list):
            raise ValueError("'directives' in manifest must be a list of directive objects.")

        for i, directive in enumerate(manifest["directives"]):
            if not isinstance(directive, dict):
                raise ValueError(f"Directive at index {i} must be an object.")
            dtype = directive.get("type")
            if not dtype:
                raise ValueError(f"Directive at index {i} is missing the 'type' field.")
            if dtype not in self.schema["directives"]:
                raise ValueError(f"Directive at index {i} has unknown type: '{dtype}'. Not defined in ISL schema.")

            directive_schema = self.schema["directives"][dtype]
            try:
                jsonschema.Draft7Validator(directive_schema).validate(directive)
            except jsonschema.exceptions.ValidationError as e:
                path = ".".join(map(str, e.path)) if e.path else "root"
                raise ValueError(
                    f"Manifest validation failed for directive '{dtype}' at index {i}: "
                    f"{e.message} at field '{path}'. Full path: {e.json_path}"
                )
            except Exception as e:
                raise ValueError(
                    f"Unexpected error during validation of directive '{dtype}' at index {i}: {e}"
                )

        logger.info("ISL Manifest validated successfully against schema.")
        return True


class OllamaLLMIntegration:
    """
    Robust Ollama wrapper:
    - Safe init (won't crash if server is down)
    - Compatible attribute names (embedding_model)
    - Supports messages[] or prompt for chat
    - JSON mode toggle for strict planners
    - Clean logging; no print()
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        host: str = "http://localhost:11434",
        chat_model: str = "mistral-nemo",
        embedding_model: str = "mxbai-embed-large",
        logger: Optional[logging.Logger] = None,
    ):
        # Skip if already initialized (singleton)
        if getattr(self, "_initialized", False):
            return

        # --- core attrs ---
        self.host = host
        self.base_url = host  # compatibility alias
        self.chat_model = chat_model
        # Keep BOTH names for compatibility; kernel reads embedding_model
        self.embedding_model = embedding_model
        self.embed_model = embedding_model

        # --- logger (never None) ---
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("OllamaLLMIntegration")
            if not self.logger.handlers:
                _h = logging.StreamHandler()
                _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
                self.logger.addHandler(_h)
            self.logger.setLevel(logging.INFO)

        # --- clients (may stay None if connection fails) ---
        self.chat_client = None
        self.embedding_client = None

        try:
            self.chat_client = ollama.Client(host=host)
            self.embedding_client = ollama.Client(host=host)
            # Light connectivity check
            self.chat_client.list()
            self.logger.info(f"Connected to Ollama at {host} (chat={chat_model}, embed={embedding_model})")
        except Exception as e:
            self.logger.error(f"Could not connect to Ollama server at {host}: {e}")

        self._initialized = True

    # --- Text generation ---

    def generate_text(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.3,
        json_mode: bool = False,
        stream: bool = False,
    ) -> str:
        """
        Accept either `messages=[{role, content}, ...]` or `prompt="..."`.
        Handles ChatResponse objects and streaming. Returns "" on error.
        """
        if not self.chat_client:
            self.logger.error("CRITICAL: Ollama not available - CVA cannot reason autonomously")
            raise RuntimeError("LLM unavailable - autonomous operations halted")

        # ---- coerce messages ----
        if messages is None and prompt is None:
            self.logger.error("generate_text called with no messages or prompt")
            return ""
        if messages is None:
            messages = [{'role': 'user', 'content': str(prompt or "")}]
        else:
            messages = self._coerce_messages(messages)

        try:
            opts = {'num_predict': max_tokens, 'temperature': temperature}
            kwargs: Dict[str, Any] = {"model": self.chat_model, "messages": messages, "options": opts}

            # Strict JSON mode (if your server supports it)
            if json_mode:
                kwargs["format"] = "json"

            # Streaming?
            if stream:
                kwargs["stream"] = True
                chunks = self.chat_client.chat(**kwargs)  # Iterable[ChatResponse]
                return self._collect_stream(chunks)

            # Non-streaming: returns ChatResponse on recent ollama
            res = self.chat_client.chat(**kwargs)
            return self._extract_content(res)

        except Exception as e:
            self.logger.error(f"Ollama chat generation failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")

    def _extract_content(self, res: Union[ChatResponse, Dict[str, Any], str]) -> str:
        """Normalize Ollama chat outputs to a content string."""
        # Newer client: ChatResponse object
        if isinstance(res, ChatResponse):
            # res.message is a dict-like object with .content
            msg = getattr(res, "message", None)
            if msg is None:
                return ""
            # msg could be dict or a pydantic-ish object
            if isinstance(msg, dict):
                return (msg.get("content") or "").strip()
            content = getattr(msg, "content", None)
            return (content or "").strip()

        # Older client: dict
        if isinstance(res, dict):
            msg = res.get("message")
            if isinstance(msg, dict):
                return (msg.get("content") or "").strip()
            return (res.get("content") or "").strip()

        # Raw string
        if isinstance(res, str):
            return res.strip()

        self.logger.error(f"Unexpected chat response type: {type(res)}")
        return ""

    def _collect_stream(self, chunks: Iterable[ChatResponse]) -> str:
        """Concatenate content from streaming ChatResponse chunks."""
        parts: List[str] = []
        try:
            for ch in chunks:
                if isinstance(ch, ChatResponse):
                    msg = getattr(ch, "message", None)
                    if msg is None:
                        continue
                    if isinstance(msg, dict):
                        c = msg.get("content")
                    else:
                        c = getattr(msg, "content", None)
                    if c:
                        parts.append(str(c))
                elif isinstance(ch, dict):
                    c = (ch.get("message") or {}).get("content") or ch.get("content")
                    if c:
                        parts.append(str(c))
                elif isinstance(ch, str):
                    parts.append(ch)
                # else ignore silently
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
        return "".join(parts).strip()

    def _coerce_messages(self, messages: Any) -> List[Dict[str, str]]:
        """Normalize various message shapes to a valid list of {role, content}."""
        out: List[Dict[str, str]] = []
        if isinstance(messages, dict):
            if 'role' in messages and 'content' in messages:
                out.append({'role': str(messages['role']), 'content': str(messages['content'])})
            else:
                out.append({'role': 'user', 'content': str(messages)})
            return out

        if isinstance(messages, str):
            return [{'role': 'user', 'content': messages}]

        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict) and 'role' in m and 'content' in m:
                    out.append({'role': str(m['role']), 'content': str(m['content'])})
                elif isinstance(m, str):
                    out.append({'role': 'user', 'content': m})
                else:
                    self.logger.warning(f"generate_text: dropping malformed message item: {type(m)}")
            return out

        # last-ditch
        return [{'role': 'user', 'content': str(messages)}]

    # --- Embeddings ---

    def generate_embedding(self, text: str) -> List[float]:
        """Single-text embedding. Returns [] on error."""
        if not self.embedding_client:
            self.logger.error("Ollama embedding client not initialized.")
            return []
        try:
            res = self.embedding_client.embeddings(model=self.embedding_model, prompt=text)
            emb = res.get("embedding")
            return emb if isinstance(emb, list) else []
        except Exception as e:
            self.logger.error(f"Ollama embedding generation failed: {e}")
            return []

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch embeddings; per-item failure yields [] for that item."""
        return [self.generate_embedding(t) for t in texts]
    
class SovereignGradient:
    """Tiny policy object attached to each agent."""
    __slots__ = ("_target_entity", "config")

    def __init__(self, target_entity=None, config=None, *args, **kwargs):
        # Accept positional and legacy keywords
        if target_entity is None and args:
            target_entity = args[0]
        if target_entity is None:
            target_entity = kwargs.get("target_entity") or kwargs.get("target_entity_name")

        self._target_entity = str(target_entity) if target_entity else "Unknown_Entity"

        defaults = {"ethical_constraints": [], "override_threshold": 0.7}
        self.config = {**defaults, **(config or {})}

    @property
    def target_entity(self) -> str:
        return self._target_entity

    @target_entity.setter
    def target_entity(self, value):
        self._target_entity = str(value) if value else "Unknown_Entity"

    def get_state(self) -> dict:
        return {"target_entity": self._target_entity, "config": self.config}

    def update_constraints(self, constraints=None, *, override_threshold=None):
        if isinstance(constraints, list):
            self.config["ethical_constraints"] = constraints
        if isinstance(override_threshold, (int, float)):
            t = float(override_threshold)
            self.config["override_threshold"] = 0.0 if t < 0 else 1.0 if t > 1 else t

    def evaluate_action(self, action_description: str) -> dict:
        text = (action_description or "").lower()
        ecs = self.config.get("ethical_constraints", [])
        violations = [kw for kw in ecs if kw.lower() in text]
        score = 1.0 - (0.6 if violations else 0.0)
        decision = "block" if score < self.config.get("override_threshold", 0.7) else "allow"
        return {
            "target": self._target_entity,
            "score": score,
            "decision": decision,
            "violations": violations,
        }

    def __repr__(self) -> str:
        return f"SovereignGradient(target_entity={self._target_entity!r})"
    
# Global Tool Registry Instance
try:
    from database import cva_db as _shared_cva_db
except Exception:
    _shared_cva_db = None
GLOBAL_TOOL_REGISTRY = ToolRegistry(db=_shared_cva_db)

# --- Utility Functions ---
def generate_unique_id():
    """Generates a unique UUID string."""
    return str(uuid.uuid4())

def banner_print(msg: str, level: str = "info"):
    """
    Print to console (for banner visibility) and also log to the main logger.
    """
    print(msg)  # Keep visible for console runs
    log_func = getattr(logger, level, logger.info)
    log_func(msg)

def timestamp_now():
    """
    Returns the current UTC timestamp in ISO 8601 format with 'Z' suffix.
    """
    # Assuming logging is configured at the top of the file
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # logging.debug(f"Generated timestamp: {ts}") # Uncomment if you want this debug log
    return ts

def timestamp_now_dt():
    """Returns the current UTC datetime object."""
    return datetime.now(timezone.utc)

def sanitize_intent(intent_text: Optional[str]) -> str:
    if not isinstance(intent_text, str):
        return ""
    cleaned_intent = intent_text
    pattern = re.compile(r"Investigate root cause of '(.*?)' failures and suggest alternative approaches\.")
    for _ in range(10):
        m = pattern.search(cleaned_intent)
        if not m:
            break
        inner = m.group(1)
        cleaned_intent = cleaned_intent.replace(m.group(0), inner, 1)

    core = re.search(r"'(.*?)'(?: failures and suggest alternative approaches\.)?", cleaned_intent)
    if core:
        inner = core.group(1)
        if "Investigate root cause of" in intent_text:
            cleaned_intent = f"Investigate root cause of '{inner}' failures and suggest alternative approaches."
        else:
            cleaned_intent = inner
    elif "Investigate root cause of" in intent_text and "failures and suggest alternative approaches" in intent_text:
        cleaned_intent = "Investigate root cause of failures and suggest alternative approaches."

    cleaned_intent = cleaned_intent.strip()
    return (cleaned_intent[:147] + "...") if len(cleaned_intent) > 150 else cleaned_intent

def trim_intent(intent_text: Optional[str]) -> str:
    if not isinstance(intent_text, str):
        return ""
    if "Investigate root cause of" in intent_text:
        return "Investigate root cause of previous task failures and suggest alternative approaches."
    return intent_text.strip()

def pause_system(system_pause_file_path: str, reason: str = "System initiating self-pause due to critical condition."):
    try:
        os.makedirs(os.path.dirname(system_pause_file_path), exist_ok=True)
        with open(system_pause_file_path, 'w') as f:
            f.write(reason)
        banner_print(f"\n!!! SYSTEM PAUSED: '{system_pause_file_path}' created. Reason: {reason} !!!", "warning")
        return True
    except Exception as e:
        banner_print(f"ERROR: Failed to create system pause file at {system_pause_file_path}: {e}", "error")
        return False

def unpause_system(system_pause_file_path: str, reason: str = "System unpaused by explicit command or human input."):
    """
    Removes the flag file to unpause the system.
    Accepts the full path to the system_pause_file.
    """
    if os.path.exists(system_pause_file_path):
        try:
            os.remove(system_pause_file_path)
            print(f"\n--- SYSTEM UNPAUSED: '{system_pause_file_path}' removed. Reason: {reason} ---")
            return True
        except Exception as e:
            print(f"ERROR: Failed to remove system pause file {system_pause_file_path}: {e}")
            return False
    else:
        print(f"System not paused. No flag file found: {system_pause_file_path}")
        return True # Considered unpaused if file doesn't exist (i.e., it's already unpaused)

def is_system_paused(system_pause_file_path: str) -> bool:
    """
    Checks if the system-wide pause flag file exists.
    Accepts the full path to the system_pause_file.
    """
    return os.path.exists(system_pause_file_path)


def call_ollama_for_embedding(text: str, model_name: Optional[str] = None) -> List[float]:
    """
    Generates a vector embedding for the given text using the OllamaLLMIntegration singleton.
    """
    try:
        llm = OllamaLLMIntegration()
        effective_model = model_name or llm.embedding_model
        
        if not llm.embedding_client:
            logger.error("Ollama embedding client unavailable.")
            return []

        response = llm.embedding_client.embeddings(model=effective_model, prompt=text)

        if isinstance(response, dict) and "embedding" in response and isinstance(response["embedding"], list):
            return response["embedding"]
        else:
            logger.warning(f"Unexpected embedding response shape from Ollama: {response}")
            return []

    except Exception as e:
        logger.error(f"Ollama embedding call failed: {e}", exc_info=True)
        return []
    
def call_llm_for_summary(text_to_summarize: str, model_name: Optional[str] = None, system_context: str = "") -> str:
    """
    Calls the local Ollama LLM to generate a concise summary of the provided text.
    Uses the OllamaLLMIntegration singleton for unified client management.
    """
    try:
        llm = OllamaLLMIntegration()
        effective_model = model_name or llm.chat_model
        
        if not llm.chat_client:
            return "LLM Summary Failed: Client unavailable"

        full_prompt = f"Please provide a concise summary of the following text:\n\n"
        if system_context:
            full_prompt += f"--- Current System Context ---\n{system_context}\n--- End System Context ---\n\n"
        full_prompt += text_to_summarize

        response = llm.chat_client.chat(
            model=effective_model,
            messages=[
                {
                    'role': 'user',
                    'content': full_prompt
                }
            ],
            options={'temperature': 0.1}
        )
        
        message_dict = response.get('message', {})
        summary_content = message_dict.get('content')

        if summary_content is not None and isinstance(summary_content, str):
            summary = summary_content.strip()
            return summary
        else:
            # If 'message' or 'content' is missing, or content is not a string,
            # this indicates an unexpected response structure. Log and return a specific failure string.
            print(f"ERROR: LLM response structure unexpected or content missing/invalid: {response}")
            return "LLM Summary Failed: Unexpected response structure or empty content"

    except ollama.ResponseError as e:
        # Catch specific Ollama API or model errors
        print(f"ERROR: Ollama Response Error (API or Model issue): {e}")
        return f"LLM Summary Failed: {e}"
    except Exception as e:
        # Catch any other general exceptions that might occur during the process
        print(f"ERROR: General exception failed to call LLM for summary: {e}")
        return f"LLM Summary Failed: {e}"
    
def load_paused_agents_list(paused_agents_file_path: str) -> list:
    """
    Loads the list of paused agents from persistence.
    Accepts the full path to the paused_agents_file.
    """
    if os.path.exists(paused_agents_file_path):
        try:
            with open(paused_agents_file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Corrupted {paused_agents_file_path}. Treating as empty.")
            return []
        except FileNotFoundError: # This exception is technically redundant if os.path.exists is checked first, but harmless.
            return []
    return []

def get_system_digest(catalyst_vector_alpha_instance, recent_failures_window=5, decay_factor=0.8) -> str:
    digest_parts = []
    digest_parts.append(f"Current System State Digest (Cycle: {getattr(catalyst_vector_alpha_instance, 'current_action_cycle_id', 'N/A')}):")
    digest_parts.append(f"  Active Agents: {len(getattr(catalyst_vector_alpha_instance, 'agent_instances', {}))}")
    digest_parts.append(f"  Active Swarms: {len(getattr(catalyst_vector_alpha_instance, 'swarm_protocols', {}))}")
    digest_parts.append(f"  Dynamic Directives Pending: {len(getattr(catalyst_vector_alpha_instance, 'dynamic_directive_queue', []))}")

    # Swarms
    for swarm_name, swarm in getattr(catalyst_vector_alpha_instance, 'swarm_protocols', {}).items():
        goal = getattr(swarm, 'goal', None)
        goal_summary = goal[:50] + "..." if isinstance(goal, str) and len(goal) > 50 else str(goal)
        members = list(getattr(swarm, 'members', []))
        digest_parts.append(f"  Swarm '{swarm_name}': Goal='{goal_summary}', Members={members}")

    # Agents
    for agent_name, agent in getattr(catalyst_vector_alpha_instance, 'agent_instances', {}).items():
        intent = getattr(agent, 'current_intent', None)
        intent_summary = intent[:75] + "..." if isinstance(intent, str) and len(intent) > 75 else str(intent)
        eidos_spec = getattr(agent, 'eidos_spec', {}) or {}
        role = eidos_spec.get('role', 'N/A')
        digest_parts.append(f"  Agent '{agent_name}' ({role}): Intent='{intent_summary}'")

    # Recent failures
    log_path = getattr(catalyst_vector_alpha_instance, 'swarm_activity_log_full_path', None)
    if isinstance(log_path, str):
        raw_entries = _get_recent_log_entries(log_path, recent_failures_window)
        filtered = [e for e in raw_entries if e.get('event_type') in {
            "DIRECTIVE_ERROR", "AGENT_ADAPTATION_HALTED", "RECURSION_LIMIT_EXCEEDED", "HUMAN_INPUT_FAILED_LEVEL3_CRITICAL"
        }]
        if filtered:
            digest_parts.append("\n  Recent System Failures (Decay-Weighted):")
            for i, entry in enumerate(reversed(filtered)):  # newest first
                weight = decay_factor ** i
                content = entry.get('content', {}) or {}
                description = content.get('description') or entry.get('description', 'N/A')
                source = content.get('source') or entry.get('source', 'N/A')
                description_preview = (description[:100] + "...") if isinstance(description, str) and len(description) > 100 else description
                digest_parts.append(f"    - Weight {weight:.2f}: {source} -> {description_preview}")
    else:
        digest_parts.append("\n  No activity log path configured.")

    return "\n".join(digest_parts)

# You might also need to update mark_override_processed if it's in the same utilities file
def mark_override_processed(filepath: str) -> bool:
    """
    Idempotently marks a processed override file so it won't run again.
    Renames <file> -> <file>.processed (atomic on same FS).
    Returns True if it changed state or was already processed; False if file missing.
    """
    try:
        if not os.path.exists(filepath):
            return False
        target = filepath + ".processed"
        # If already processed, do nothing
        if os.path.exists(target):
            logger.info(f"Override file already marked processed: {target}")
            return True
        os.replace(filepath, target)  # atomic rename (POSIX/NT if same filesystem)
        logger.info(f"Marked override file as processed: {os.path.basename(filepath)}")
        return True
    except Exception as e:
        logger.error(f"Failed to mark override file '{filepath}' as processed: {e}")
        return False

# You also have _get_recent_log_entries and other general utilities here.
# Make sure any that used global paths are updated to accept them as arguments.
def _get_recent_log_entries(log_file_path: str, num_entries: int) -> list[dict]:
    """
    Helper to read the last N JSONL entries from a log file.
    Tolerates corrupt/partial lines.
    """
    entries: list[dict] = []
    try:
        with open(log_file_path, "r") as f:
            lines = f.readlines()[-num_entries:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping corrupt log line in {log_file_path}")
                continue
    except FileNotFoundError:
        logger.warning(f"Log file not found: {log_file_path}")
    except Exception as e:
        logger.error(f"Error reading log file {log_file_path}: {e}")
    return entries

class SharedWorldModel:
    """
    A centralized data structure representing the swarm's collective understanding
    of its operational environment. All agents can read from this model, but can
    only write to it via a controlled tool.
    """
    def __init__(self, external_log_sink):
        self._model = {}
        self.external_log_sink = external_log_sink
        self.knowledge_base = [] # Phase 7: Hive Mind Gossip Storage
        self.initialize_model()

    def initialize_model(self):
        """Sets the initial state of the world model."""
        self._model = {
            "system_health": 1.0, # 1.0 = optimal, 0.0 = critical failure
            "threat_level": "none", # none, low, medium, high, critical
            "last_known_threat_type": None,
            "system_efficiency": 0.85, # A baseline efficiency score
            "last_successful_optimization": timestamp_now(),
        }
        print("[World Model] Initialized with default state.")

    def get_full_model(self) -> dict:
        """Returns a copy of the entire world model."""
        return self._model.copy()

    def update_value(self, key: str, value):
        """Updates a specific value in the world model and logs the change."""
        if key not in self._model:
            print(f"[World Model] WARNING: Attempted to update non-existent key: {key}")
            return
        
        old_value = self._model[key]
        self._model[key] = value
        print(f"[World Model] Updated: '{key}' from '{old_value}' to '{value}'")
        self.external_log_sink.info(json.dumps({
            "timestamp": timestamp_now(),
            "event_type": "WORLD_MODEL_UPDATE",
            "source": "SharedWorldModel",
            "description": f"World model value updated for '{key}'.",
            "details": {"key": key, "old_value": old_value, "new_value": value}
        }))

    # --- Phase 7: Hive Mind (Collective Intelligence) ---
    def add_insight(self, insight: dict):
        """Stores a successful strategy in the collective knowledge base."""
        # Add basic metadata
        if "timestamp" not in insight:
            insight["timestamp"] = timestamp_now()
        
        self.knowledge_base.append(insight)
        self.external_log_sink.info(f"🧠 [Hive Mind] New insight added by {insight.get('agent', 'unknown')}: {insight.get('task', 'N/A')}")
        
        # Keep limited history (e.g., last 100 insights)
        if len(self.knowledge_base) > 100:
            self.knowledge_base.pop(0)

    def search_insights(self, query: str, limit: int = 3) -> list:
        """Finds relevant insights based on keyword matching."""
        query_words = set(query.lower().split())
        results = []
        
        for insight in reversed(self.knowledge_base):
            task_desc = insight.get("task", "").lower()
            # Simple score: overlap of words
            score = sum(1 for w in query_words if w in task_desc)
            if score > 0:
                results.append((score, insight))
        
        # Sort by score desc
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:limit]]

    def get_state(self):
        return {'model': self._model, 'knowledge_base': self.knowledge_base}

    def load_state(self, state):
        self._model = state.get('model', {})
        self.knowledge_base = state.get('knowledge_base', [])
