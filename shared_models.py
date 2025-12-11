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

@dataclass
class BusMessage:
    sender: str
    recipient: str
    message_type: str
    content: any
    task_description: Optional[str]
    status: str
    cycle_id: str
    timestamp: str


class MessageBus:
    def __init__(self):
        self.messages = {}
        self.lock = Lock()
        self.catalyst_vector_ref = None

    def send_message(self, sender: str, recipient: str, message_type: str, content: any, 
                     task_description: str = None, status: str = "pending", cycle_id: str = None):
        """Thread-safe message sending."""
        with self.lock:
            if recipient not in self.messages:
                self.messages[recipient] = []
            
            self.messages[recipient].append({
                "sender": sender,
                "message_type": message_type,
                "content": content,
                "task_description": task_description,
                "status": status,
                "cycle_id": cycle_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    def get_messages_for(self, agent_name: str):
        """Thread-safe message retrieval - matches what agents expect."""
        with self.lock:
            msgs = self.messages.get(agent_name, [])
            self.messages[agent_name] = []
            return msgs
        
    def send_directive(self, directive):
        if self.catalyst_vector_ref:
            enqueue_fn = getattr(self.catalyst_vector_ref, "enqueue_directive", None)
            if enqueue_fn:
                enqueue_fn(directive)
            else:
                # Hard fail to avoid unsynchronized writes
                raise RuntimeError("enqueue_directive not available on catalyst_vector_ref")

class EventMonitor:
    def __init__(self):
        self.event_history = []
        self.agent_responses = defaultdict(list) # This is the defaultdict
        self.current_cycle_id = None
        self._lock = Lock()

    def set_current_cycle(self, cycle_id: str):
        with self._lock:
            self.current_cycle_id = cycle_id

    def log_event(self, event_type: str, event_id: str, payload: dict):
        event_record = {
            'event_id': event_id,
            'type': event_type,
            'urgency': payload.get('urgency'),
            'change_factor': payload.get('change_factor'),
            'direction': payload.get('direction'),
            'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'cycle_id': self.current_cycle_id
        }
        with self._lock:
            self.event_history.append(event_record)
        print(f"  [EventMonitor] Logged event: {event_record['event_id'][:8]} ({event_record['type']})")

    def log_agent_response(self, agent_id: str, event_id: str, response_type: str, details: dict = None):
        response_record = {
            'event_id': event_id,
            'response_type': response_type,
            'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'details': details if details is not None else {},
            'cycle_id': self.current_cycle_id
        }
        with self._lock:
            self.agent_responses[agent_id].append(response_record)
        print(f"  [EventMonitor] Agent {agent_id} responded to {event_id[:8]} ({response_type})")

    def get_event_history(self, event_id: str = None):
        with self._lock:
            if event_id:
                return [e for e in self.event_history if e['event_id'] == event_id]
            return list(self.event_history)

    def get_agent_event_responses(self, agent_id: str, event_id: str = None):
        with self._lock:
            responses = list(self.agent_responses.get(agent_id, []))
        if event_id:
            return [r for r in responses if r['event_id'] == event_id]
        return responses

    def get_state(self):
        """Returns the current state of the EventMonitor for persistence."""
        with self._lock:
            serializable_agent_responses = {
                agent_id: list(responses) for agent_id, responses in self.agent_responses.items()
            }
            return {
                'event_history': list(self.event_history),
                'agent_responses': serializable_agent_responses,
                'current_cycle_id': self.current_cycle_id
            }

    def load_state(self, state):
        """Loads the state into the EventMonitor."""
        with self._lock:
            self.event_history = state.get('event_history', [])
            loaded_responses = state.get('agent_responses', {})
            self.agent_responses = defaultdict(list, {
                k: list(v) for k, v in loaded_responses.items() # Ensure values are lists for defaultdict
            })
            self.current_cycle_id = state.get('current_cycle_id', None)

# ==================================================================
#  2. Agent Memory & Cognition
# ==================================================================
# ChromaDB is optional; we degrade gracefully if it's not installed/available.
try:
    import chromadb  # type: ignore
except Exception:
    chromadb = None  # allows the class to continue without vector store


def timestamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

class MemeticKernel:
    """
    Manages an agent's memory, including active deque and optional ChromaDB archiving.

    `llm_integration` interface expected (duck-typed):
      - summarize(text: str, model_name: str) -> str
      - generate_embedding(text: str) -> List[float]
    """

    def __init__(
        self,
        agent_name: str,
        llm_integration: Any,
        external_log_sink: logging.Logger,
        chroma_db_path: str,
        persistence_dir: str,
        config: Optional[dict] = None,
        loaded_memories: Optional[list] = None,
        memetic_archive_path: Optional[str] = None,
    ):
        self.agent_name = agent_name
        self.llm_integration = llm_integration
        self.external_log_sink = external_log_sink

        # Ensure config is always a dictionary
        self.config = config if isinstance(config, dict) else {}

        # Simple in-memory DB for any keyed stores you want
        self.memory_db = collections.defaultdict(list)

        # Primary active memories (deque for recent, in-memory events)
        initial_memories_for_deque: List[dict] = []
        if (
            loaded_memories
            and isinstance(loaded_memories, collections.abc.Iterable)
            and not isinstance(loaded_memories, (str, bytes, dict))
        ):
            initial_memories_for_deque = [m for m in loaded_memories if isinstance(m, dict)]

        effective_maxlen = self.config.get("max_memory_length", 100)
        if not isinstance(effective_maxlen, int) or effective_maxlen <= 0:
            effective_maxlen = 100

        self.memories = collections.deque(initial_memories_for_deque, maxlen=effective_maxlen)
        self.external_log_sink.info(
            f"MemeticKernel for {self.agent_name} initialized 'memories' deque (maxlen={effective_maxlen}). "
            f"Initial count: {len(self.memories)}",
            extra={
                "agent": self.agent_name,
                "deque_maxlen": effective_maxlen,
                "initial_memory_count": len(self.memories),
            },
        )

        self.compressed_memories = collections.deque(maxlen=10)
        self.current_cycle_ref = 0
        self.compression_paused_until_cycle = 0
        self.is_compression_paused = False
        self.last_received_message_summary = None

        # --- Local file-based archiving paths ---
        self.persistence_dir = persistence_dir
        self.memetic_archive_path = (
            memetic_archive_path
            if memetic_archive_path
            else os.path.join(self.persistence_dir, f"memetic_archive_{self.agent_name}.jsonl")
        )
        self.log_file = os.path.join(self.persistence_dir, f"memetic_log_{self.agent_name}.jsonl")

        os.makedirs(os.path.dirname(self.memetic_archive_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._initialize_log()

        # --- Initialize ChromaDB client (optional) ---
        self.chroma_db_full_path = chroma_db_path
        self.chroma_client = None
        self.compressed_memories_collection = None
        try:
            if chromadb is None:
                raise RuntimeError("chromadb not available")

            os.makedirs(self.chroma_db_full_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_full_path)  # type: ignore[attr-defined]
            collection_name = f"agent-{self.agent_name.lower().replace('_', '-')}-memories"
            self.compressed_memories_collection = self.chroma_client.get_or_create_collection(name=collection_name)
            self.external_log_sink.info(
                f"Connected to ChromaDB for {self.agent_name}. Collection: '{collection_name}'.",
                extra={"agent": self.agent_name, "db_path": self.chroma_db_full_path},
            )
        except Exception as e:
            self.external_log_sink.critical(
                f"Failed to initialize ChromaDB for {self.agent_name}: {e}",
                exc_info=True,
                extra={"agent": self.agent_name, "error": str(e), "db_path": self.chroma_db_full_path},
            )

    # --------------------------- helpers ---------------------------

    def _safe_summarize(self, text: str, model_name: str) -> str:
        """Prefer llm_integration; fall back to a simple heuristic summarizer."""
        try:
            if hasattr(self.llm_integration, "summarize"):
                summary = self.llm_integration.summarize(text, model_name=model_name)
                if summary and str(summary).strip():
                    return summary
        except Exception as e:
            self.external_log_sink.error(
                f"llm_integration.summarize errored: {e}", exc_info=True, extra={"agent": self.agent_name}
            )
        # naive fallback (first 1000 chars with basic compaction)
        compact = " ".join(text.split())
        return compact[:1000]

    def _safe_embed(self, text: str) -> Optional[List[float]]:
        """Prefer llm_integration; if unavailable, return None (we'll file-archive only)."""
        try:
            if hasattr(self.llm_integration, "generate_embedding"):
                emb = self.llm_integration.generate_embedding(text)
                if emb and isinstance(emb, list):
                    return emb
        except Exception as e:
            self.external_log_sink.error(
                f"Embedding generation errored: {e}", exc_info=True, extra={"agent": self.agent_name}
            )
        return None

    def _archive_compressed_memory(self, compressed_memory: dict):
        """
        Archives a single compressed memory entry to ChromaDB.
        Raises RuntimeError on failure (caller handles fallback).
        """
        if not self.compressed_memories_collection:
            error_msg = f"ChromaDB collection not initialized for {self.agent_name}. Cannot archive memory."
            self.external_log_sink.error(error_msg, extra={"agent": self.agent_name})
            raise RuntimeError(error_msg)

        try:
            doc_id = f"comp-mem-{compressed_memory['timestamp']}-{random.randint(0, 9999)}"
            self.compressed_memories_collection.add(
                documents=[compressed_memory["summary"]],
                embeddings=[compressed_memory["embedding"]],
                metadatas=[
                    {
                        "timestamp": compressed_memory["timestamp"],
                        "agent_name": self.agent_name,
                        "original_memory_count": compressed_memory["original_memory_count"],
                        "type": compressed_memory["type"],  # "CompressedMemory"
                    }
                ],
                ids=[doc_id],
            )
            print(f"  [MemeticKernel] Archived compressed memory {doc_id} to ChromaDB.")
            self.external_log_sink.info(
                f"Archived compressed memory {doc_id} to ChromaDB.",
                extra={"agent": self.agent_name, "doc_id": doc_id},
            )
        except Exception as e:
            error_msg = (
                f"ChromaDB archiving failed for {self.agent_name} "
                f"(ID: {doc_id if 'doc_id' in locals() else 'N/A'}): {e}"
            )
            self.external_log_sink.error(
                error_msg,
                extra={
                    "agent": self.agent_name,
                    "error": str(e),
                    "compressed_memory_preview": str(compressed_memory)[:100],
                },
            )
            raise RuntimeError(error_msg)

    def _initialize_log(self):
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, "w"):
                    pass
            except IOError as e:
                print(f"ERROR: Could not initialize log file {self.log_file} for {self.agent_name}: {e}")

    def _log_memory(self, memory: dict):
        """Appends a raw memory entry to the agent's local memetic log file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(memory) + "\n")
        except Exception as e:
            self.external_log_sink.error(
                f"Failed to write to local memory log for {self.agent_name}: {e}",
                extra={"agent": self.agent_name, "error": str(e), "memory_type": memory.get("type")},
            )

    def _save_raw_memory_to_file_archive(self, memory_entry: dict):
        """Appends a memory entry to the local memetic archive JSONL."""
        try:
            with open(self.memetic_archive_path, "a") as f:
                f.write(json.dumps(memory_entry) + "\n")
        except Exception as e:
            self.external_log_sink.error(
                f"Failed to write raw memory to file archive for {self.agent_name}: {e}",
                extra={"agent": self.agent_name, "error": str(e)},
            )

    # --------------------------- public API ---------------------------

    def add_memory(
        self,
        memory_type: str,
        content: Any,
        timestamp: Optional[str] = None,
        related_event_id: Optional[str] = None,
        task_id: Optional[str] = None,
        source_agent: Optional[str] = None,
    ):
        """
        Adds a new memory to the agent's memory stream, now accepting source_agent.
        """
        if timestamp is None:
            timestamp_str = timestamp_now()
        elif isinstance(timestamp, str):
            timestamp_str = timestamp
        else:
            self.external_log_sink.warning(
                f"MemeticKernel: Invalid timestamp format provided for memory type '{memory_type}'. "
                f"Expected str, got {type(timestamp).__name__}. Using current timestamp. "
                f"Content preview: {str(content)[:100]}",
                extra={"agent": source_agent if source_agent else self.agent_name},
            )
            timestamp_str = timestamp_now()

        memory = {
            "timestamp": timestamp_str,
            "type": memory_type,
            "content": content,
            "related_event_id": related_event_id,
            "task_id": task_id,
            "cycle_id": self.current_cycle_ref,
            "source_agent": source_agent or self.agent_name,
        }

        self.memories.append(memory)
        self._log_memory(memory)
        print(f"  [MemeticKernel] {memory.get('source_agent', self.agent_name)}: Stored {memory_type} memory.")

    def inhibit_compression(self, cycles: int):
        """Pauses memory compression for a specified number of cycles."""
        self.compression_paused_until_cycle = self.current_cycle_ref + cycles
        self.is_compression_paused = True
        print(f"  [MemeticKernel] Compression paused for {self.agent_name} until cycle {self.compression_paused_until_cycle}.")
        self.add_memory(
            "CompressionPause",
            {"until_cycle": self.compression_paused_until_cycle, "reason": "Explicit inhibition"},
        )

    def get_recent_memories(self, limit: int = 10) -> list:
        """Retrieves the most recent memories, up to a given limit."""
        return list(self.memories)[-limit:]

    def query_long_term_memory(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.compressed_memories_collection:
            self.external_log_sink.error(
                f"LTM query aborted for {self.agent_name}: vector store unavailable.", extra={"agent": self.agent_name}
            )
            return []

        self.external_log_sink.info(
            f"{self.agent_name} querying LTM", extra={"agent": self.agent_name, "query_preview": query_text[:120]}
        )
        try:
            query_embedding = self._safe_embed(query_text)
            if not query_embedding:
                self.external_log_sink.warning(
                    "Embedding generation failed; skipping LTM query.",
                    extra={"agent": self.agent_name},
                )
                return []

            results = self.compressed_memories_collection.query(
                query_embeddings=[query_embedding], n_results=top_k
            )

            hits: List[Dict[str, Any]] = []
            docs = results.get("documents") or [[]]
            metas = results.get("metadatas") or [[]]
            dists = results.get("distances") or [[]]
            for i, doc in enumerate(docs[0]):
                meta = metas[0][i] if i < len(metas[0]) else {}
                dist = dists[0][i] if i < len(dists[0]) else None
                hits.append(
                    {
                        "summary": doc,
                        "timestamp": meta.get("timestamp"),
                        "relevance_score": dist,
                    }
                )

            self.external_log_sink.info(
                f"LTM query complete: {len(hits)} hits",
                extra={"agent": self.agent_name, "results_found": len(hits)},
            )
            return hits
        except Exception as e:
            self.external_log_sink.error(
                f"LTM query errored: {e}", exc_info=True, extra={"agent": self.agent_name}
            )
            return []

    def get_timestamp(self, memory: dict) -> str:
        """Extract and validate the timestamp from a memory object."""
        ts = memory.get("timestamp", "1970-01-01T00:00:00Z")
        if isinstance(ts, dict):
            self.external_log_sink.warning(
                f"Unexpected dict timestamp in memory: {memory}", extra={"agent": self.agent_name}
            )
            return "1970-01-01T00:00:00Z"
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return ts
        except ValueError:
            self.external_log_sink.warning(
                f"Invalid timestamp format in memory: {memory}", extra={"agent": self.agent_name}
            )
            return "1970-01-01T00:00:00Z"

    def summarize_and_compress_memories(
        self,
        memories_to_process: list,
        model_name: str = "llama3",
        embedding_model: str = "nomic-embed-text",  # kept for compatibility; not used directly
    ) -> bool:
        """
        Summarizes a batch of raw memories, generates a vector embedding for the summary,
        and archives the compressed memory. Returns True on success (even if vector-store is offline).
        """
        if not memories_to_process:
            print(f"  [MemeticKernel] No memories to process for {self.agent_name}.")
            return False

        # Check if compression is currently inhibited based on current_cycle_ref
        if self.current_cycle_ref < self.compression_paused_until_cycle:
            print(
                f"  [MemeticKernel] Compression for {self.agent_name} is temporarily paused "
                f"(resumes after cycle {self.compression_paused_until_cycle}). Skipping this cycle."
            )
            self.external_log_sink.debug(
                f"Compression skipped for {self.agent_name} due to active pause.",
                extra={"agent": self.agent_name, "paused_until": self.compression_paused_until_cycle},
            )
            return True

        # Prepare content for summarization
        contents_to_summarize: List[str] = []
        for m in memories_to_process:
            mem_content = m.get("content")
            mem_type = m.get("type")

            if isinstance(mem_content, str):
                contents_to_summarize.append(mem_content)
            elif isinstance(mem_content, dict):
                if mem_type == "CompressedMemory":
                    contents_to_summarize.append(mem_content.get("summary", str(mem_content)))
                elif mem_content.get("summary"):
                    contents_to_summarize.append(mem_content["summary"])
                elif mem_content.get("task") and mem_content.get("outcome"):
                    contents_to_summarize.append(
                        f"Task: {mem_content['task']}, Outcome: {mem_content.get('outcome', 'N/A')}"
                    )
                elif mem_content.get("new_intent"):
                    contents_to_summarize.append(f"Intent Adapted to: {mem_content['new_intent']}")
                elif mem_content.get("tool_name"):
                    contents_to_summarize.append(
                        f"Tool Used: {mem_content['tool_name']}, Output: {str(mem_content.get('tool_output', 'N/A'))[:50]}"
                    )
                elif mem_content.get("patterns"):
                    contents_to_summarize.append(f"Patterns: {str(mem_content['patterns'])[:100]}")
                elif mem_content.get("message"):
                    contents_to_summarize.append(f"Message: {mem_content['message']}")
                else:
                    contents_to_summarize.append(json.dumps(mem_content))
            else:
                contents_to_summarize.append(str(mem_content))

        concatenated_content = "\n".join(contents_to_summarize).strip()
        if not concatenated_content:
            print(f"  [MemeticKernel] No substantial content to summarize for {self.agent_name}.")
            self.external_log_sink.debug(
                f"No substantial content for {self.agent_name} to compress.", extra={"agent": self.agent_name}
            )
            return False

        print(
            f"  [MemeticKernel] {self.agent_name} initiating LLM summary for {len(memories_to_process)} memories..."
        )

        summary = self._safe_summarize(concatenated_content, model_name=model_name)
        if not summary or not str(summary).strip():
            print(f"  [MemeticKernel] LLM summarization failed or returned empty for {self.agent_name}.")
            self.external_log_sink.error(
                f"LLM summarization failed or empty for {self.agent_name}.",
                extra={"agent": self.agent_name},
            )
            return False

        print(f"  [MemeticKernel] {self.agent_name} generating embedding for summary...")
        embedding = self._safe_embed(summary)

        compressed_memory_entry = {
            "timestamp": timestamp_now(),
            "type": "CompressedMemory",
            "summary": summary,
            "embedding": embedding,  # may be None if vector-store/embeddings unavailable
            "original_memory_count": len(memories_to_process),
            "source_memories_preview": [
                (m.get("content", "")[:50] if isinstance(m.get("content"), str) else str(m.get("content"))[:50])
                for m in memories_to_process[:3]
            ],
        }

        # Keep a short local deque of compressed summaries
        self.compressed_memories.append(compressed_memory_entry)

        # Try to archive to Chroma if both store & embedding exist; otherwise archive to file
        if self.compressed_memories_collection and embedding is not None:
            try:
                self._archive_compressed_memory(compressed_memory_entry)
            except RuntimeError:
                self._save_raw_memory_to_file_archive({"type": "CompressedMemory", **compressed_memory_entry})
        else:
            # No vector store or no embedding -> file-only archive
            self._save_raw_memory_to_file_archive({"type": "CompressedMemory", **compressed_memory_entry})

        # ✅ PRUNE ORIGINALS SAFELY (prevents re-compressing the same items forever)
        try:
            count_to_prune = min(len(memories_to_process), len(self.memories))
            for _ in range(count_to_prune):
                self.memories.popleft()
        except Exception as e:
            self.external_log_sink.warning(
                "Pruning originals after compression failed: %s",
                e,
                extra={"agent": self.agent_name, "to_prune": len(memories_to_process)},
            )

        # ✅ Always return True once we've produced a compressed entry (even if vector store was unavailable)
        return True

    def retrieve_recent_memories(self, lookback_period: int = 20) -> list:
        """Retrieves a specified number of recent memories from the local kernel (deque)."""
        if hasattr(self, "memories") and isinstance(self.memories, collections.deque):
            return list(self.memories)[-lookback_period:]
        else:
            self.external_log_sink.warning(
                f"[MemeticKernel] Warning: 'memories' deque not initialized or is not a deque for agent {self.agent_name}. Returning empty list.",
                extra={"agent": self.agent_name},
            )
            return []

    def update_last_received_message(self, message: Any):
        if not hasattr(self, "message_history"):
            self.message_history = []
        self.message_history.append(message)
        self.add_memory(
            "LastMessageReceived", {"message_preview": str(message)[:100], "agent": self.agent_name}
        )

    def store_memory(self, memory_type: str, content: Any):
        """Store a memory with a timestamp in the memory database (wrapper calling add_memory)."""
        if not hasattr(self, "memory_db"):
            self.memory_db = collections.defaultdict(list)

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.add_memory(memory_type, content, timestamp=ts)
        self.external_log_sink.debug(f"[{self.agent_name}] Stored memory in DB: {memory_type} - {str(content)[:100]}...")

    def clear_working_memory(self):
        """
        Clears the agent's current working memory/context (the deque).
        Retains compressed memories.
        """
        self.memories.clear()
        print(f"  [MemeticKernel] {self.agent_name}: Working memory/context has been cleared.")
        self.add_memory(
            "MemoryReset",
            {"reason": "Working memory cleared by swarm reset protocol", "agent": self.agent_name},
        )

    def get_state(self) -> dict:
        """Returns the current state of the MemeticKernel for persistence."""
        return {
            "config": self.config,
            "memories": list(self.memories),
            "compressed_memories": list(self.compressed_memories),
            "last_received_message_summary": self.last_received_message_summary,
            "compression_paused_until_cycle": self.compression_paused_until_cycle,
            "is_compression_paused": self.is_compression_paused,
            "memetic_archive_path": self.memetic_archive_path,
        }

    def load_state(self, state: dict):
        """Restores the kernel's state from a saved dictionary."""
        if not state:
            return

        self.config = state.get("config", self.config)

        loaded_memories = state.get("memories", [])
        max_len = self.config.get("max_memory_length", 100)
        self.memories = collections.deque(loaded_memories, maxlen=max_len)

        self.memetic_archive_path = state.get("memetic_archive_path", self.memetic_archive_path)
        self.next_compression_cycle = state.get("next_compression_cycle", 1)

        print(f"  [MemeticKernel] {self.agent_name}: State restored. Loaded {len(self.memories)} memories.")

    def reflect(self) -> str:
        """
        Synthesizes a detailed self-narrative from the agent's memories,
        prioritizing recent raw memories, compressed insights, and specific event types.
        """
        if not self.memories and not self.compressed_memories:
            return "My journey includes: No memories yet."

        reflection_points: List[str] = []

        memories_for_reflection: List[dict] = list(self.memories)
        memories_for_reflection.extend(list(self.compressed_memories))
        memories_for_reflection.sort(key=self.get_timestamp)

        lookback_count = 10
        start_index = max(0, len(memories_for_reflection) - lookback_count)

        for memory in memories_for_reflection[start_index:]:
            memory_type = memory.get("type")
            mem_content = memory.get("content") if memory_type != "CompressedMemory" else memory

            if mem_content is None:
                reflection_points.append(f"[{memory.get('timestamp', 'N/A')}][InvalidMem] Missing content/type.")
                continue

            if isinstance(mem_content, dict):
                if memory_type == "TaskOutcome":
                    task_name = mem_content.get("task", "N/A")
                    outcome_status = mem_content.get("outcome", "N/A")
                    reflection_points.append(f"[{memory['timestamp']}][TaskOutcome] Task: '{task_name}', Outcome: {outcome_status}.")
                elif memory_type == "LLMSummary":
                    original_task_preview = mem_content.get("original_task", "N/A")[:50] + "..."
                    summary_preview = mem_content.get("summary", "")[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][LLMSummary] Summarized: {original_task_preview}, Preview: {summary_preview}.")
                elif memory_type == "PlanningSuccess":
                    goal_preview = mem_content.get("goal", "N/A")[:50] + "..."
                    success_type = mem_content.get("type", "N/A")
                    reflection_points.append(f"[{memory['timestamp']}][PlanSuccess] Goal: '{goal_preview}' (Type: {success_type}).")
                elif memory_type == "PlanningKnowledgeStored":
                    goal_preview = mem_content.get("goal", "N/A")[:50] + "..."
                    directives_count = mem_content.get("directives_count", "N/A")
                    source = mem_content.get("source", "N/A")
                    reflection_points.append(f"[{memory['timestamp']}][PlanKBStored] Goal: '{goal_preview}', Directives: {directives_count}, Source: {source}.")
                elif memory_type == "PlanningKnowledgeRetrieved":
                    goal_preview = mem_content.get("goal", "N/A")[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][PlanKBRetrieved] Goal: '{goal_preview}'.")
                elif memory_type == "PlanningFallback":
                    reason_preview = mem_content.get("reason", "N/A")[:50] + "..."
                    goal_preview = mem_content.get("goal", "N/A")[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][PlanFallback] Reason: '{reason_preview}', Goal: '{goal_preview}'.")
                elif memory_type == "FailureAnalysis":
                    task_failed_preview = mem_content.get("task_failed", "N/A")[:50] + "..."
                    analysis_summary_preview = mem_content.get("analysis_summary", "N/A")[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][FailureAnalysis] Failed Task: '{task_failed_preview}', Analysis: {analysis_summary_preview}.")
                elif memory_type == "SolutionFound":
                    problem_addressed_preview = mem_content.get("problem_addressed", "N/A")[:50] + "..."
                    solution_summary_preview = mem_content.get("solution_summary", "N/A")[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][SolutionFound] Problem: '{problem_addressed_preview}', Summary: {solution_summary_preview}.")
                elif memory_type == "CompressedMemory":
                    summary_preview = mem_content.get("summary", "")[:50] + "..."
                    original_count = mem_content.get("original_memory_count", "N/A")
                    reflection_points.append(f"[{memory['timestamp']}][Comp.Mem] {summary_preview} (from {original_count} originals).")
                elif memory_type == "PlanningOutcome":
                    task_preview = mem_content.get("task", "N/A")[:50] + "..."
                    outcome_status = mem_content.get("outcome", "N/A")
                    reflection_points.append(f"[{memory['timestamp']}][PlanOutcome] Goal: '{task_preview}', Outcome: {outcome_status}.")
                elif memory_type == "PatternInsight":
                    patterns_preview = ""
                    if mem_content.get("patterns") and isinstance(mem_content["patterns"], list):
                        first_pattern_obj = mem_content["patterns"][0] if mem_content["patterns"] else ""
                        if isinstance(first_pattern_obj, str):
                            patterns_preview = first_pattern_obj.strip()
                            if "LLM Insight:" in patterns_preview:
                                patterns_preview = patterns_preview.split("LLM Insight:", 1)[1].strip()
                            if len(patterns_preview) > 70:
                                patterns_preview = patterns_preview[:70] + "..."
                        else:
                            patterns_preview = str(first_pattern_obj)[:70] + "..."
                    else:
                        patterns_preview = "No specific patterns content."
                    reflection_points.append(f"[{memory['timestamp']}][Patt.Insight] {patterns_preview}.")
                elif memory_type == "MessageSent":
                    recipient = mem_content.get("recipient", "N/A")
                    msg_type = mem_content.get("type", "N/A")
                    preview = mem_content.get("preview", "N/A")
                    reflection_points.append(f"[{memory['timestamp']}][MsgSent] To: {recipient}, Type: {msg_type}, Preview: {preview}.")
                elif memory_type == "CompressionPause":
                    reason = mem_content.get("reason", "N/A")
                    paused_until = mem_content.get("paused_until_cycle", "N/A")
                    reflection_points.append(f"[{memory['timestamp']}][CompPause] Reason: {reason}, Paused Until Cycle: {paused_until}.")
                elif memory_type in ("event", "InjectedEvent"):
                    event_type_name = mem_content.get("type", mem_content.get("event_type", "N/A"))
                    payload = mem_content.get("payload", {})
                    if isinstance(payload, dict):
                        urgency = payload.get("urgency", "N/A")
                        direction = payload.get("direction", "N/A")
                        change_factor = payload.get("change_factor", "N/A")
                        payload_summary = f"Urgency: {urgency}, Dir: {direction}, Change: {change_factor}"
                    else:
                        payload_summary = str(payload)[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][Event] Type: {event_type_name}, Payload: {payload_summary}.")
                elif memory_type == "CommandReceived":
                    command_type = mem_content.get("command_type", "N/A")
                    params_preview = str(mem_content.get("command_params", {}))[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][CmdRecv] Type: {command_type}, Params: {params_preview}.")
                elif memory_type == "HumanInputAcknowledged":
                    response_preview = mem_content.get("response", "N/A")[:50] + "..."
                    context_preview = mem_content.get("context", "N/A")[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][HumanAck] Response: {response_preview}, Context: {context_preview}.")
                else:
                    reflection_points.append(f"[{memory['timestamp']}][UnknownDictMem] {str(mem_content)[:50]}...")
            else:
                if memory_type == "Activation":
                    reflection_points.append(
                        f"[{memory['timestamp']}][Activation] {mem_content}. Current intent: '{self.config.get('current_intent', 'N/A')}';"
                    )
                elif memory_type == "SwarmReportSummary":
                    reflection_points.append(f"[{memory['timestamp']}][SwarmReportSummary] {mem_content}.")
                elif memory_type == "IntentAdaptation":
                    reflection_points.append(f"[{memory['timestamp']}][IntentAdaptation] {mem_content}.")
                elif memory_type == "IntentAdaptationWarning":
                    reflection_points.append(f"[{memory['timestamp']}][IntentAdaptationWarning] {mem_content}.")
                elif memory_type == "FallbackIntent":
                    reflection_points.append(f"[{memory['timestamp']}][FallbackIntent] {mem_content}.")
                elif memory_type == "SelfTransformation":
                    reflection_points.append(f"[{memory['timestamp']}][SelfTransformation] {mem_content}.")
                elif memory_type == "IntentAlignment":
                    reflection_points.append(f"[{memory['timestamp']}][IntentAlignment] {mem_content}.")
                elif memory_type == "IntentNonAlignment":
                    reflection_points.append(f"[{memory['timestamp']}][IntentNonAlignment] {mem_content}.")
                elif memory_type == "DiagnosticReport":
                    reflection_points.append(f"[{memory['timestamp']}][DiagnosticReport] {mem_content}.")
                elif memory_type == "SwarmFormation":
                    reflection_points.append(f"[{memory['timestamp']}][SwarmFormation] {mem_content}.")
                elif memory_type == "MemberAdded":
                    reflection_points.append(f"[{memory['timestamp']}][MemberAdded] {mem_content}.")
                elif memory_type == "GoalUpdate":
                    reflection_points.append(f"[{memory['timestamp']}][GoalUpdate] {mem_content}.")
                elif memory_type == "GradientUpdate":
                    reflection_points.append(f"[{memory['timestamp']}][GradientUpdate] {mem_content}.")
                elif memory_type == "TaskCoordination":
                    reflection_points.append(f"[{memory['timestamp']}][TaskCoordination] {mem_content}.")
                elif memory_type == "PlannerInitialization":
                    reflection_points.append(f"[{memory['timestamp']}][PlannerInit] {mem_content}.")
                elif memory_type == "SelfReboot":
                    reflection_points.append(f"[{memory['timestamp']}][SelfReboot] {mem_content}.")
                elif memory_type == "CriticalSelfDiagnosisRecursion":
                    reflection_points.append(f"[{memory['timestamp']}][CritDiagRec] {mem_content}.")
                else:
                    reflection_points.append(f"[{memory['timestamp']}][UnknownStringMem] {mem_content}.")

        final_narrative = f"My journey includes: {' '.join(reflection_points)}"

        # Truncated console/log output
        max_reflect_len = 500
        truncated = (final_narrative[:max_reflect_len] + "...") if len(final_narrative) > max_reflect_len else final_narrative
        print(f"  [MemeticKernel] {self.agent_name} reflects: '{truncated}'")
        return final_narrative

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

    def __init__(
        self,
        host: str = "http://localhost:11434",
        chat_model: str = "mistral-small",
        embedding_model: str = "mxbai-embed-large",
        logger: Optional[logging.Logger] = None,
    ):
        # --- core attrs ---
        self.host = host
        self.base_url = host  # compatibility alias
        self.chat_model = chat_model
        from shared_models import OllamaLLMIntegration
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


def call_ollama_for_embedding(text: str, model_name: str = "nomic-embed-text") -> List[float]:
    """
    Generates a vector embedding for the given text using the specified Ollama model.
    This version is updated to correctly handle the response structure from the ollama library.
    """
    try:
        client = ollama.Client(host='http://localhost:11434')
        
        # The ollama library's embeddings function returns a dictionary.
        # The actual embedding vector is inside the 'embedding' key.
        response = client.embeddings(model=model_name, prompt=text)

        if isinstance(response, dict) and "embedding" in response and isinstance(response["embedding"], list):
            return response["embedding"]
        else:
            # This handles cases where the response might be in an unexpected format
            logger.warning(f"Unexpected embedding response shape from Ollama: {response}")
            return []

    except Exception as e:
        print(f"ERROR: Failed to call Ollama for embedding with model '{model_name}': {e}")
        logger.error(f"Ollama embedding call failed for model '{model_name}': {e}", exc_info=True)
        return []
    
def call_llm_for_summary(text_to_summarize: str, model_name: str = "llama3", system_context: str = "") -> str:
    """
    Calls the local Ollama LLM to generate a concise summary of the provided text.
    Uses the synchronous Ollama client, now including system context.
    Ensures robust handling of the LLM response structure.
    """
    try:
        client = ollama.Client(host='http://localhost:11434') # Default Ollama host
        print(f"[IP-Integration] Executed Microsoft™-aligned runtime for local LLM inference via {model_name} for summarization.")

        full_prompt = f"Please provide a concise summary of the following text:\n\n"
        if system_context:
            full_prompt += f"--- Current System Context ---\n{system_context}\n--- End System Context ---\n\n"
        full_prompt += text_to_summarize

        response = client.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': full_prompt
                }
            ],
            options={'temperature': 0.1} # Lower temperature for factual summary
        )
        
        # --- CRITICAL FIX: Defensively access keys in the LLM response ---
        # Get the 'message' dictionary from the response. Default to an empty dictionary if 'message' key is missing.
        message_dict = response.get('message', {})
        
        # Get the 'content' from the message_dict. Default to None if 'content' key is missing.
        summary_content = message_dict.get('content')

        # Check if summary_content is not None and is a string
        if summary_content is not None and isinstance(summary_content, str):
            summary = summary_content.strip()
            print(f"[LLM] Summarized text using {model_name}.")
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

    def get_state(self):
        return {'model': self._model}

    def load_state(self, state):
        self._model = state.get('model', {})
