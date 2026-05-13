"""
MemeticKernel — extracted from shared_models.py for clarity.

Manages an agent's memory stream: active deque, ChromaDB archiving,
LLM-powered compression, and self-reflection.
"""
from __future__ import annotations

import collections
import collections.abc
import json
import logging
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
