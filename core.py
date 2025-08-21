# ==================================================================
#  core.py - Core Components for Catalyst Vector Alpha
# ==================================================================

# --- Standard Library Imports ---
import logging
import json
import os
import random
import uuid
import collections
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Optional, List, Dict

# --- Third-Party Library Imports ---
import yaml
import ollama
import chromadb
import jsonschema
import psutil
import re
from tools import (
    get_system_cpu_load_tool,
    initiate_network_scan_tool,
    deploy_recovery_protocol_tool,
    update_resource_allocation_tool,
    get_environmental_data_tool,
)

# --- Helper Functions ---
def timestamp_now() -> str:
    """Returns a UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()

# --- Globals used across the module ---
system_instance = None
main_app_logger = None

# --- Logging Setup (clean + idempotent) ---
logger = logging.getLogger("CatalystLogger")
logger.setLevel(logging.DEBUG)
logger.propagate = False

for h in list(logger.handlers):
    logger.removeHandler(h)

root_logger = logging.getLogger()
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)

_console = logging.StreamHandler()
_console.setLevel(logging.DEBUG)
_console.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "%Y-%m-%d %H:%M:%S"
))
logger.addHandler(_console)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(ch)


# ==================================================================
#  1. Communication & Event Handling
# ==================================================================

class MessageBus:
    """A central message queue for inter-agent communication."""
    def __init__(self):
        self.messages = deque()
        self.catalyst_vector_ref = None
        self.current_cycle_id = "initialization"

    def send_message(self, sender: str, recipient: str, message_type: str, content: any, 
                     task_description: str = None, status: str = "pending", cycle_id: str = None):
        """Sends a structured message from one component to another."""
        self.messages.append({
            "sender": sender, "recipient": recipient,
            "message_type": message_type, "content": content,
            "task_description": task_description, "status": status,
            "cycle_id": cycle_id or self.current_cycle_id,
            "timestamp": timestamp_now()
        })

    def get_messages_for_agent(self, agent_name: str) -> List[Dict]:
        """Retrieves and clears all messages for a specific agent."""
        agent_messages = [msg for msg in self.messages if msg["recipient"] == agent_name]
        self.messages = deque([msg for msg in self.messages if msg["recipient"] != agent_name])
        return agent_messages

class EventMonitor:
    def __init__(self):
        self.event_history = []
        self.agent_responses = defaultdict(list) # This is the defaultdict
        self.current_cycle_id = None

    def set_current_cycle(self, cycle_id: str):
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
        self.agent_responses[agent_id].append(response_record)
        print(f"  [EventMonitor] Agent {agent_id} responded to {event_id[:8]} ({response_type})")

    def get_event_history(self, event_id: str = None):
        if event_id:
            return [e for e in self.event_history if e['event_id'] == event_id]
        return self.event_history

    def get_agent_event_responses(self, agent_id: str, event_id: str = None):
        responses = self.agent_responses.get(agent_id, [])
        if event_id:
            return [r for r in responses if r['event_id'] == event_id]
        return responses

    def get_state(self):
        """Returns the current state of the EventMonitor for persistence."""
        # FIX: Convert defaultdict to dict for serialization
        serializable_agent_responses = {
            agent_id: list(responses) for agent_id, responses in self.agent_responses.items()
        }
        return {
            'event_history': self.event_history,
            'agent_responses': serializable_agent_responses,
            'current_cycle_id': self.current_cycle_id
        }

    def load_state(self, state):
        """Loads the state into the EventMonitor."""
        self.event_history = state.get('event_history', [])
        # FIX: Load agent_responses back into a defaultdict
        loaded_responses = state.get('agent_responses', {})
        self.agent_responses = defaultdict(list, {
            k: list(v) for k, v in loaded_responses.items() # Ensure values are lists for defaultdict
        })
        self.current_cycle_id = state.get('current_cycle_id', None)

# ==================================================================
#  2. Agent Memory & Cognition
# ==================================================================

class MemeticKernel:
    """Manages an agent's memory, including active deque and ChromaDB archiving."""
    def __init__(self, agent_name: str, external_log_sink: logging.Logger,
                 chroma_db_path: str,
                 persistence_dir: str,
                 config: Optional[dict] = None,
                 loaded_memories: Optional[list] = None, # Can be a list of dicts
                 memetic_archive_path: Optional[str] = None):

        self.agent_name = agent_name
        self.external_log_sink = external_log_sink # Passed from ProtoAgent

        # Ensure config is always a dictionary
        if isinstance(config, dict):
           self.config = config
        elif config is not None:
            try:
                import json
                self.config = json.loads(config) if isinstance(config, str) else dict(config)
            except Exception:
                self.config = {}
                self.external_log_sink.warning(
                    f"MemeticKernel for {self.agent_name} received invalid config type ({type(config)}). Using empty config.",
                    extra={"agent": self.agent_name or "Unknown", "config_type": str(type(config))}
                )
        else:
            self.config = {}

        self.memory_db = collections.defaultdict(list) # For categorized memory storage (if used)

        # Primary active memories (deque for recent, in-memory events)
        # Ensure loaded_memories is always an iterable before passing to deque
        initial_memories_for_deque = []
        if loaded_memories is None:
            initial_memories_for_deque = []
        # Check for common non-iterable types that might slip through, e.g. a single dict, not a list of dicts.
        elif not isinstance(loaded_memories, collections.abc.Iterable) or isinstance(loaded_memories, (str, bytes, dict)):
            self.external_log_sink.error(
                f"MemeticKernel for {self.agent_name} received loaded_memories that is not a proper iterable sequence (type: {type(loaded_memories)}). Initializing with empty memories.",
                extra={"agent": self.agent_name, "loaded_memories_type": str(type(loaded_memories))}
            )
            initial_memories_for_deque = []
        else:
            # Validate loaded_memories list content before using
            loaded_memories = [
                m for m in loaded_memories
                if isinstance(m, dict) and "timestamp" in m and "content" in m
            ]
            initial_memories_for_deque = loaded_memories

        # Use a default maxlen from config, or fallback to a hardcoded value if config is not present or invalid
        effective_maxlen = self.config.get('max_memory_length', 100)
        if not isinstance(effective_maxlen, int) or effective_maxlen <= 0:
            self.external_log_sink.warning(
                f"Invalid max_memory_length '{effective_maxlen}' in config for {self.agent_name}. Defaulting to 100.",
                extra={"agent": self.agent_name, "config_maxlen": effective_maxlen}
            )
            effective_maxlen = 100

        self.memories = collections.deque(initial_memories_for_deque, maxlen=effective_maxlen)
        self.external_log_sink.info(
            f"MemeticKernel for {self.agent_name} initialized 'memories' deque (maxlen={effective_maxlen}). Initial count: {len(self.memories)}",
            extra={"agent": self.agent_name, "deque_maxlen": effective_maxlen, "initial_memory_count": len(self.memories)}
        )
        
        # This deque is explicitly initialized and should not be an issue
        self.compressed_memories = collections.deque(maxlen=10) 

        self.current_cycle_ref = 0 
        self.compression_paused_until_cycle = 0 
        self.is_compression_paused = False 

        self.last_received_message_summary = None

        # --- Local file-based archiving paths ---
        self.persistence_dir = persistence_dir 
        self.memetic_archive_path = memetic_archive_path if memetic_archive_path else \
                                    os.path.join(self.persistence_dir, f"memetic_archive_{self.agent_name}.jsonl")
        self.log_file = os.path.join(self.persistence_dir, f"memetic_log_{self.agent_name}.jsonl")

        # Ensure the directories for local log files exist
        os.makedirs(os.path.dirname(self.memetic_archive_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)


        # --- Initialize ChromaDB client ---
        self.chroma_db_full_path = chroma_db_path 
        try:
            # Ensure the ChromaDB directory exists
            os.makedirs(self.chroma_db_full_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_full_path)

            collection_name = f"agent-{self.agent_name.lower().replace('_', '-')}-memories"
            self.compressed_memories_collection = self.chroma_client.get_or_create_collection(name=collection_name)
            self.external_log_sink.info(f"Connected to ChromaDB for {self.agent_name}. Collection: '{collection_name}'.", extra={"agent": self.agent_name, "db_path": self.chroma_db_full_path})
        except Exception as e:
            self.external_log_sink.critical(f"Failed to initialize ChromaDB for {self.agent_name}: {e}", exc_info=True, extra={"agent": self.agent_name, "error": str(e), "db_path": self.chroma_db_full_path})
            self.chroma_client = None
            self.compressed_memories_collection = None

    def _archive_compressed_memory(self, compressed_memory: dict):
        """
        Archives a single compressed memory entry to ChromaDB.
        """
        if not self.compressed_memories_collection:
            # This is where MEMORY_COMPRESSION_FAILED might originate if ChromaDB init failed silently
            error_msg = f"ChromaDB collection not initialized for {self.agent_name}. Cannot archive memory."
            self.external_log_sink.error(error_msg, extra={"agent": self.agent_name})
            raise RuntimeError(error_msg)

        try:
            # ChromaDB requires IDs to be strings
            doc_id = f"comp-mem-{compressed_memory['timestamp']}-{random.randint(0, 9999)}"

            # Add the compressed memory to the ChromaDB collection
            self.compressed_memories_collection.add(
                documents=[compressed_memory['summary']],
                embeddings=[compressed_memory['embedding']],
                metadatas=[{
                    "timestamp": compressed_memory['timestamp'],
                    "agent_name": self.agent_name,
                    "original_memory_count": compressed_memory['original_memory_count'],
                    "type": compressed_memory['type'] # "CompressedMemory"
                }],
                ids=[doc_id]
            )
            print(f"  [MemeticKernel] Archived compressed memory {doc_id} to ChromaDB.")
            self.external_log_sink.info(f"Archived compressed memory {doc_id} to ChromaDB.", extra={"agent": self.agent_name, "doc_id": doc_id})
        except Exception as e:
            error_msg = f"ChromaDB archiving failed for {self.agent_name} (ID: {doc_id if 'doc_id' in locals() else 'N/A'}): {e}"
            self.external_log_sink.error(error_msg, extra={"agent": self.agent_name, "error": str(e), "compressed_memory_preview": str(compressed_memory)[:100]})
            raise RuntimeError(error_msg) # Re-raise to be caught by caller (summarize_and_compress_memories)
    
    def get_recent_memories(self, limit: int = 10) -> list:
        """Retrieves the most recent memories, up to a given limit."""
        return list(self.memories)[-limit:]
    
    # The _initialize_log method for local file, keep as is
    def _initialize_log(self):
        # Uses self.log_file which is constructed with self.agent_name
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'w') as f:
                    pass # Create empty file
            except IOError as e:
                print(f"ERROR: Could not initialize log file {self.log_file} for {self.agent_name}: {e}")

    def add_memory(self, memory_type: str, content: any,
                   timestamp: Optional[str]=None,
                   related_event_id: Optional[str]=None,
                   task_id: Optional[str]=None,
                   source_agent: Optional[str]=None): # <--- ADD THIS PARAMETER
        """
        Adds a new memory to the agent's memory stream, now accepting source_agent.
        """
        if timestamp is None:
            timestamp_str = timestamp_now()
        elif isinstance(timestamp, str):
            timestamp_str = timestamp
        else:
            self.external_log_sink.warning(
                f"MemeticKernel: Invalid timestamp format provided for memory type '{memory_type}'. Expected str, got {type(timestamp).__name__}. Using current timestamp. Content preview: {str(content)[:100]}",
                extra={"agent": source_agent if source_agent else self.agent_name}
            )
            timestamp_str = timestamp_now()

        memory = {
            "timestamp": timestamp_str,
            "type": memory_type,
            "content": content,
            "related_event_id": related_event_id,
            "task_id": task_id,
            "cycle_id": self.current_cycle_ref # Assuming this is correctly set by the main system loop
        }

        # --- NEW LOGIC: Add source_agent to memory if provided, else use kernel's agent_name ---
        if source_agent:
            memory["source_agent"] = source_agent
        else:
            memory["source_agent"] = self.agent_name # Fallback to the agent associated with this kernel instance

        self.memories.append(memory) # This line is correct as `self.memories`

        self._log_memory(memory) # Ensure this method logs the full `memory` dict
        # Updated print statement for clarity
        print(f"  [MemeticKernel] {memory.get('source_agent', self.agent_name)}: Stored {memory_type} memory.")

    def _log_memory(self, memory):
        """Appends a raw memory entry to the agent's local memetic log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(memory) + '\n')
        except Exception as e:
            self.external_log_sink.error(f"Failed to write to local memory log for {self.agent_name}: {e}", extra={"agent": self.agent_name, "error": str(e), "memory_type": memory.get('type')})

    def inhibit_compression(self, cycles: int):
        """
        Pauses memory compression for a specified number of cycles.
        Called by ProtoAgent.receive_event.
        """
        self.compression_paused_until_cycle = self.current_cycle_ref + cycles
        self.is_compression_paused = True # Set the flag when inhibited
        print(f"  [MemeticKernel] Compression paused for {self.agent_name} until cycle {self.compression_paused_until_cycle}.")
        self.add_memory("CompressionPause", {"until_cycle": self.compression_paused_until_cycle, "reason": "Explicit inhibition"})


    def get_timestamp(self, memory):
        """
        Extract and validate the timestamp from a memory object.
        Returns a string timestamp or a default value if invalid.
        """
        timestamp = memory.get('timestamp', '1970-01-01T00:00:00Z')
        if isinstance(timestamp, dict): # Handle cases where content might be dict but interpreted as timestamp
            self.external_log_sink.warning(f"Unexpected dict timestamp in memory: {memory}", extra={"agent": self.agent_name})
            return '1970-01-01T00:00:00Z'
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return timestamp
        except ValueError:
            self.external_log_sink.warning(f"Invalid timestamp format in memory: {memory}", extra={"agent": self.agent_name})
            return '1970-01-01T00:00:00Z'

    def summarize_and_compress_memories(self, memories_to_process: list, model_name="llama3", embedding_model="nomic-embed-text"):
        """
        Summarizes a batch of raw memories, generates a vector embedding for the summary,
        and archives the compressed memory.
        """
        if not memories_to_process:
            print(f"  [MemeticKernel] No memories to process for {self.agent_name}.")
            return False

        # Check if compression is currently inhibited based on current_cycle_ref
        if self.current_cycle_ref < self.compression_paused_until_cycle:
            print(f"  [MemeticKernel] Compression for {self.agent_name} is temporarily paused (resumes after cycle {self.compression_paused_until_cycle}). Skipping this cycle.")
            self.external_log_sink.debug(f"Compression skipped for {self.agent_name} due to active pause.", extra={"agent": self.agent_name, "paused_until": self.compression_paused_until_cycle})
            return True

        # Assume call_llm_for_summary and call_ollama_for_embedding are available globally or imported.
        # Example: from .utilities import call_llm_for_summary, call_ollama_for_embedding

        contents_to_summarize = []
        for m in memories_to_process:
            mem_content = m.get('content') # Use .get for safety
            mem_type = m.get('type')

            if isinstance(mem_content, str):
                contents_to_summarize.append(mem_content)
            elif isinstance(mem_content, dict):
                if mem_type == 'CompressedMemory': # Special handling if a compressed memory somehow ended up in raw
                    contents_to_summarize.append(mem_content.get('summary', str(mem_content)))
                elif mem_content.get('summary'):
                    contents_to_summarize.append(mem_content['summary'])
                elif mem_content.get('task') and mem_content.get('outcome'):
                    contents_to_summarize.append(f"Task: {mem_content['task']}, Outcome: {mem_content.get('outcome', 'N/A')}")
                elif mem_content.get('new_intent'):
                    contents_to_summarize.append(f"Intent Adapted to: {mem_content['new_intent']}")
                elif mem_content.get('tool_name'):
                    contents_to_summarize.append(f"Tool Used: {mem_content['tool_name']}, Output: {mem_content.get('tool_output', 'N/A')[:50]}")
                elif mem_content.get('patterns'):
                    contents_to_summarize.append(f"Patterns: {str(mem_content['patterns'])[:100]}")
                elif mem_content.get('message'):
                    contents_to_summarize.append(f"Message: {mem_content['message']}")
                else: # Fallback for other dicts, convert to string
                    contents_to_summarize.append(json.dumps(mem_content)) # Use json.dumps for dicts
            else:
                contents_to_summarize.append(str(mem_content)) # Fallback for non-str/non-dict types

        concatenated_content = "\n".join(contents_to_summarize)
        if not concatenated_content.strip():
            print(f"  [MemeticKernel] No substantial content to summarize for {self.agent_name}.")
            self.external_log_sink.debug(f"No substantial content for {self.agent_name} to compress.", extra={"agent": self.agent_name})
            return False

        print(f"  [MemeticKernel] {self.agent_name} initiating LLM summary for {len(memories_to_process)} memories...")
        # Assume call_llm_for_summary is correctly defined and accessible.
        # It needs to return a string summary.
        summary = call_llm_for_summary(concatenated_content, model_name=model_name)

        if not summary or "LLM Summary Failed" in summary: # Check for empty summary too
            print(f"  [MemeticKernel] LLM summarization failed or returned empty for {self.agent_name}.")
            self.external_log_sink.error(f"LLM summarization failed or empty for {self.agent_name}.", extra={"agent": self.agent_name, "summary_error_msg": summary[:100] if summary else "Empty summary"})
            return False

        print(f"  [MemeticKernel] {self.agent_name} generating embedding for summary...")
        # Assume call_ollama_for_embedding is correctly defined and accessible.
        # It needs to return an embedding (list of floats).
        embedding = call_ollama_for_embedding(summary, model_name=embedding_model)

        if not embedding:
            print(f"  [MemeticKernel] Embedding generation failed for {self.agent_name}.")
            self.external_log_sink.error(f"Embedding generation failed for {self.agent_name}.", extra={"agent": self.agent_name, "summary_preview": summary[:100]})
            return False

        compressed_memory_entry = { # Renamed to compressed_memory_entry to avoid conflict with local variable in try block
            "timestamp": timestamp_now(),
            "type": "CompressedMemory",
            "summary": summary,
            "embedding": embedding,
            "original_memory_count": len(memories_to_process),
            "source_memories_preview": [
                (m['content'][:50] if isinstance(m['content'], str) else str(m['content'])[:50]) # Handles non-string content
                for m in memories_to_process[:3]
            ]
        }
        
        # --- Store in self.compressed_memories deque (for recent compressed) ---
        self.compressed_memories.append(compressed_memory_entry)


        # --- CRITICAL: Call the archiving method to ChromaDB ---
        try:
            # Call the ChromaDB archiving method
            self._archive_compressed_memory(compressed_memory_entry)
            print(f"  [MemeticKernel] {self.agent_name} successfully compressed and archived {len(memories_to_process)} memories.")
            return True
        except RuntimeError as e: # Catch RuntimeError specifically from _archive_compressed_memory
            print(f"  [MemeticKernel ERROR] Failed to archive compressed memory for {self.agent_name}: {e}")
            # Error already logged by _archive_compressed_memory
            return False # Return False if archiving fails


    def retrieve_recent_memories(self, lookback_period: int = 20) -> list:
        """
        Retrieves a specified number of recent memories from the local kernel (deque).
        Ensures consistency for pattern detection logic.
        """
        # CRITICAL FIX: Changed from self.local_memories to self.memories
        if hasattr(self, 'memories') and isinstance(self.memories, deque): # Check for deque type
            # Convert deque to list for slicing to avoid modifying deque directly if not needed elsewhere
            return list(self.memories)[-lookback_period:]
        else:
            self.external_log_sink.warning(f"[MemeticKernel] Warning: 'memories' deque not initialized or is not a deque for agent {self.agent_name}. Returning empty list.", extra={"agent": self.agent_name})
            return []

    # Note: You have two _archive_compressed_memory methods in your original snippet.
    # The one above is for ChromaDB. The one below seems to be for file-based archiving.
    # It's better to have clear names if both are used, e.g., _archive_to_chromadb and _archive_to_file.
    # If the one below is intended for the local memetic_archive_path, keep it separate.
    # The one called by summarize_and_compress_memories is `_archive_compressed_memory` (ChromaDB version).
    # If you intend to also save compressed memories to `memetic_archive_path`, you need to call this explicitly.
    # For now, I'll assume ChromaDB is the primary "archiving" for compressed memories.

    # This seems like a helper for local memory file, distinct from ChromaDB archiving
    def _save_raw_memory_to_file_archive(self, memory_entry):
        """Appends a raw memory entry to the agent's local memetic archive file (e.g., for full history)."""
        try:
            with open(self.memetic_archive_path, 'a') as f:
                f.write(json.dumps(memory_entry) + '\n')
        except Exception as e:
            self.external_log_sink.error(f"Failed to write raw memory to file archive for {self.agent_name}: {e}", extra={"agent": self.agent_name, "error": str(e)})


    def update_last_received_message(self, message):
        # You need to decide if message_history is part of MemeticKernel's state,
        # or if it's part of the MessageBus/Agent directly.
        # If it belongs to MemeticKernel, it should be initialized in __init__
        if not hasattr(self, 'message_history'):
            self.message_history = [] # Initialize if not exists (less ideal than __init__)
        self.message_history.append(message)
        # Log this memory or process it further
        self.add_memory("LastMessageReceived", {"message_preview": str(message)[:100], "agent": self.agent_name})
        
    def store_memory(self, memory_type, content):
        """
        Store a memory with a timestamp in the memory database.
        This method is now a wrapper that calls add_memory.
        """
        # Ensure memory_db is initialized (though __init__ should handle this)
        if not hasattr(self, 'memory_db'):
            self.memory_db = defaultdict(list)
        
        # Create a memory dictionary with a timestamp
        memory = {
            'type': memory_type,
            'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'content': content
        }        
        # Call the dedicated add_memory to append to the deque and log locally
        # This prevents duplication if both store_memory and add_memory were called for the same purpose.
        self.add_memory(memory_type, content, timestamp=memory['timestamp']) # Pass content and other relevant fields

        self.external_log_sink.debug(f"[{self.agent_name}] Stored memory in DB: {memory_type} - {str(content)[:100]}...")

    def clear_working_memory(self):
        """
        Clears the agent's current working memory/context (the deque).
        This is typically called during a system reset (e.g., swarm reset)
        to refresh the agent's immediate perception and prevent old context
        from influencing new decisions. It retains compressed memories.
        """
        # CRITICAL FIX: Changed from self.local_memories to self.memories
        self.memories.clear() # Clear the deque
        print(f"  [MemeticKernel] {self.agent_name}: Working memory/context has been cleared.")
        self.add_memory("MemoryReset", {"reason": "Working memory cleared by swarm reset protocol", "agent": self.agent_name})

    def get_state(self):
        """
        Returns the current state of the MemeticKernel for persistence.
        Ensures deque is converted to a list for serialization.
        """
        return {
            'config': self.config,
            'memories': list(self.memories), # Convert deque to list for serialization
            'compressed_memories': list(self.compressed_memories), # Convert deque to list for serialization
            'last_received_message_summary': self.last_received_message_summary,
            'compression_paused_until_cycle': self.compression_paused_until_cycle,
            'is_compression_paused': self.is_compression_paused
        }

    def load_state(self, state):
        """
        Loads the state into the MemeticKernel from a dictionary.
        """
        self.config.update(state.get('config', {}))
        # Load deque from list
        self.memories = deque(state.get('memories', []), maxlen=100)
        self.compressed_memories = deque(state.get('compressed_memories', []), maxlen=10)

        self.last_received_message_summary = state.get('last_received_message_summary', None)
        self.compression_paused_until_cycle = state.get('compression_paused_until_cycle', 0)
        self.is_compression_paused = state.get('is_compression_paused', False) # Load this flag

    def reflect(self) -> str:
        """
        Synthesizes a detailed self-narrative from the agent's memories,
        prioritizing recent raw memories, compressed insights, and specific event types.
        This provides a comprehensive internal view of the agent's recent cognitive journey.
        All known memory types are explicitly formatted for readability.
        """
        # CRITICAL FIX: Changed from self.local_memories to self.memories
        if not self.memories and not self.compressed_memories:
            return f"My journey includes: No memories yet."

        reflection_points = []
        
        # Combine memories deque and compressed memories deque for reflection
        memories_for_reflection = list(self.memories) # Use self.memories (the deque)
        memories_for_reflection.extend(list(self.compressed_memories)) # Convert deque to list before extend

        # Sort memories chronologically using the robust get_timestamp method
        memories_for_reflection.sort(key=self.get_timestamp)

        # Reflect on the last few relevant memories (e.g., last 10 for a concise summary)
        lookback_count = 10 
        start_index = max(0, len(memories_for_reflection) - lookback_count)
        
        for memory in memories_for_reflection[start_index:]:
            memory_type = memory.get('type')
            
            # For 'CompressedMemory', the content is the memory object itself, not nested under 'content'
            mem_content = memory.get('content') if memory_type != 'CompressedMemory' else memory

            if mem_content is None:
                reflection_points.append(f"[{memory.get('timestamp', 'N/A')}][InvalidMem] Missing content/type.")
                continue
            
            # --- Handle Dictionary Content Types ---
            if isinstance(mem_content, dict):
                if memory_type == 'TaskOutcome':
                    task_name = mem_content.get('task', 'N/A')
                    outcome_status = mem_content.get('outcome', 'N/A')
                    reflection_points.append(f"[{memory['timestamp']}][TaskOutcome] Task: '{task_name}', Outcome: {outcome_status}.")
                
                elif memory_type == 'LLMSummary':
                    original_task_preview = mem_content.get('original_task', 'N/A')[:50] + "..."
                    summary_preview = mem_content.get('summary', '')[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][LLMSummary] Summarized: {original_task_preview}, Preview: {summary_preview}.")
                
                elif memory_type == 'PlanningSuccess':
                    goal_preview = mem_content.get('goal', 'N/A')[:50] + "..."
                    success_type = mem_content.get('type', 'N/A') # e.g., CBRRetrieval, LLMDecomposition
                    reflection_points.append(f"[{memory['timestamp']}][PlanSuccess] Goal: '{goal_preview}' (Type: {success_type}).")
                
                elif memory_type == 'PlanningKnowledgeStored':
                    goal_preview = mem_content.get('goal', 'N/A')[:50] + "..."
                    directives_count = mem_content.get('directives_count', 'N/A')
                    source = mem_content.get('source', 'N/A')
                    reflection_points.append(f"[{memory['timestamp']}][PlanKBStored] Goal: '{goal_preview}', Directives: {directives_count}, Source: {source}.")
                
                elif memory_type == 'PlanningKnowledgeRetrieved':
                    goal_preview = mem_content.get('goal', 'N/A')[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][PlanKBRetrieved] Goal: '{goal_preview}'.")
                
                elif memory_type == 'PlanningFallback':
                    reason_preview = mem_content.get('reason', 'N/A')[:50] + "..."
                    goal_preview = mem_content.get('goal', 'N/A')[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][PlanFallback] Reason: '{reason_preview}', Goal: '{goal_preview}'.")
                
                elif memory_type == 'FailureAnalysis':
                    task_failed_preview = mem_content.get('task_failed', 'N/A')[:50] + "..."
                    analysis_summary_preview = mem_content.get('analysis_summary', 'N/A')[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][FailureAnalysis] Failed Task: '{task_failed_preview}', Analysis: {analysis_summary_preview}.")
                
                elif memory_type == 'SolutionFound':
                    problem_addressed_preview = mem_content.get('problem_addressed', 'N/A')[:50] + "..."
                    solution_summary_preview = mem_content.get('solution_summary', 'N/A')[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][SolutionFound] Problem: '{problem_addressed_preview}', Summary: {solution_summary_preview}.")
                
                elif memory_type == 'CompressedMemory':
                    summary_preview = mem_content.get('summary', '')[:50] + "..."
                    original_count = mem_content.get('original_memory_count', 'N/A')
                    reflection_points.append(f"[{memory['timestamp']}][Comp.Mem] {summary_preview} (from {original_count} originals).")
                
                elif memory_type == 'PlanningOutcome':
                    task_preview = mem_content.get('task', 'N/A')[:50] + "..."
                    outcome_status = mem_content.get('outcome', 'N/A')
                    reflection_points.append(f"[{memory['timestamp']}][PlanOutcome] Goal: '{task_preview}', Outcome: {outcome_status}.")
                
                elif memory_type == 'PatternInsight':
                    patterns_preview = ""
                    if mem_content.get('patterns') and isinstance(mem_content['patterns'], list):
                        first_pattern_obj = mem_content['patterns'][0] if mem_content['patterns'] else ""
                        if isinstance(first_pattern_obj, str):
                            if "LLM Insight:" in first_pattern_obj:
                                patterns_preview = first_pattern_obj.split("LLM Insight:", 1)[1].strip()
                            else:
                                patterns_preview = first_pattern_obj.strip()
                            patterns_preview = patterns_preview[:70] + "..." if len(patterns_preview) > 70 else patterns_preview
                        else:
                            patterns_preview = str(first_pattern_obj)[:70] + "..."
                    else:
                        patterns_preview = "No specific patterns content."
                    reflection_points.append(f"[{memory['timestamp']}][Patt.Insight] {patterns_preview}.")
                
                elif memory_type == 'MessageSent': 
                    recipient = mem_content.get('recipient', 'N/A')
                    msg_type = mem_content.get('type', 'N/A')
                    preview = mem_content.get('preview', 'N/A')
                    reflection_points.append(f"[{memory['timestamp']}][MsgSent] To: {recipient}, Type: {msg_type}, Preview: {preview}.")
                
                elif memory_type == 'CompressionPause': 
                    reason = mem_content.get('reason', 'N/A')
                    paused_until = mem_content.get('paused_until_cycle', 'N/A')
                    reflection_points.append(f"[{memory['timestamp']}][CompPause] Reason: {reason}, Paused Until Cycle: {paused_until}.")

                elif memory_type == 'event' or memory_type == 'InjectedEvent': 
                    event_type_name = mem_content.get('type', mem_content.get('event_type', 'N/A'))
                    payload_summary = ""
                    payload = mem_content.get('payload', {})
                    if isinstance(payload, dict):
                        urgency = payload.get('urgency', 'N/A')
                        direction = payload.get('direction', 'N/A')
                        change_factor = payload.get('change_factor', 'N/A')
                        payload_summary = f"Urgency: {urgency}, Dir: {direction}, Change: {change_factor}"
                    else:
                        payload_summary = str(payload)[:50] + "..."

                    reflection_points.append(f"[{memory['timestamp']}][Event] Type: {event_type_name}, Payload: {payload_summary}.")

                elif memory_type == 'CommandReceived': 
                    command_type = mem_content.get('command_type', 'N/A')
                    params_preview = str(mem_content.get('command_params', {}))[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][CmdRecv] Type: {command_type}, Params: {params_preview}.")
                
                elif memory_type == 'HumanInputAcknowledged': # Add this for human input acknowledgements
                    response_preview = mem_content.get('response', 'N/A')[:50] + "..."
                    context_preview = mem_content.get('context', 'N/A')[:50] + "..."
                    reflection_points.append(f"[{memory['timestamp']}][HumanAck] Response: {response_preview}, Context: {context_preview}.")

                # --- Catch-all for any other dictionary content not explicitly handled ---
                else:
                    reflection_points.append(f"[{memory['timestamp']}][UnknownDictMem] {str(mem_content)[:50]}...")

            # --- Handle String Content Types ---
            else: 
                if memory_type == 'Activation':
                    reflection_points.append(f"[{memory['timestamp']}][Activation] {mem_content}. Current intent: '{self.config.get('current_intent', 'N/A')}';")
                elif memory_type == 'SwarmReportSummary':
                    reflection_points.append(f"[{memory['timestamp']}][SwarmReportSummary] {mem_content}.")
                elif memory_type == 'IntentAdaptation':
                    reflection_points.append(f"[{memory['timestamp']}][IntentAdaptation] {mem_content}.")
                elif memory_type == 'IntentAdaptationWarning':
                    reflection_points.append(f"[{memory['timestamp']}][IntentAdaptationWarning] {mem_content}.")
                elif memory_type == 'FallbackIntent':
                    reflection_points.append(f"[{memory['timestamp']}][FallbackIntent] {mem_content}.")
                elif memory_type == 'SelfTransformation':
                    reflection_points.append(f"[{memory['timestamp']}][SelfTransformation] {mem_content}.")
                elif memory_type == 'IntentAlignment':
                    reflection_points.append(f"[{memory['timestamp']}][IntentAlignment] {mem_content}.")
                elif memory_type == 'IntentNonAlignment':
                    reflection_points.append(f"[{memory['timestamp']}][IntentNonAlignment] {mem_content}.")
                elif memory_type == 'DiagnosticReport':
                    reflection_points.append(f"[{memory['timestamp']}][DiagnosticReport] {mem_content}.")
                elif memory_type == 'SwarmFormation':
                    reflection_points.append(f"[{memory['timestamp']}][SwarmFormation] {mem_content}.")
                elif memory_type == 'MemberAdded':
                    reflection_points.append(f"[{memory['timestamp']}][MemberAdded] {mem_content}.")
                elif memory_type == 'GoalUpdate':
                    reflection_points.append(f"[{memory['timestamp']}][GoalUpdate] {mem_content}.")
                elif memory_type == 'GradientUpdate':
                    reflection_points.append(f"[{memory['timestamp']}][GradientUpdate] {mem_content}.")
                elif memory_type == 'TaskCoordination':
                    reflection_points.append(f"[{memory['timestamp']}][TaskCoordination] {mem_content}.")
                elif memory_type == 'PlannerInitialization':
                    reflection_points.append(f"[{memory['timestamp']}][PlannerInit] {mem_content}.")
                elif memory_type == 'SelfReboot':
                    reflection_points.append(f"[{memory['timestamp']}][SelfReboot] {mem_content}.")
                elif memory_type == 'CriticalSelfDiagnosisRecursion':
                    reflection_points.append(f"[{memory['timestamp']}][CritDiagRec] {mem_content}.")
                
                # --- Catch-all for any other string content not explicitly handled ---
                else:
                    reflection_points.append(f"[{memory['timestamp']}][UnknownStringMem] {mem_content}.")
        
        # Final narrative construction and truncation for display
        final_narrative = f"My journey includes: {' '.join(reflection_points)}"
        
        # Truncate the final narrative for console/log output if too long
        max_reflect_len = 500 
        truncated_narrative_for_print = (final_narrative[:max_reflect_len] + "...") if len(final_narrative) > max_reflect_len else final_narrative
        
        # Print the truncated version for console/log (as per your existing pattern)
        print(f"  [MemeticKernel] {self.agent_name} reflects: '{truncated_narrative_for_print}'")
        
        # Return the potentially longer, full narrative for LLM input (as per distill_self_narrative)
        return final_narrative   


# ==================================================================
#  3. Tools and Tool Registry
# ==================================================================

# --- Tool Functions ---
def get_system_cpu_load_tool():
    try:
        return f"Current system CPU load is {psutil.cpu_percent(interval=1)}%."
    except Exception as e:
        return f"Tool failed: {e}"

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

GET_SYSTEM_CPU_LOAD_PARAMS = {
    "type": "object",
    "properties": {
        "time_interval_seconds": {
            "type": "integer",
            "description": "The time interval in seconds to average CPU load over. Defaults to 1 second.",
            "default": 1
        }
    },
    "required": []
}

INITIATE_NETWORK_SCAN_PARAMS = {
    "type": "object",
    "properties": {
        "target_ip": {
            "type": "string",
            "description": "The IP address or hostname to scan (e.g., '192.168.1.1', 'example.com')."
        },
        "scan_type": {
            "type": "string",
            "description": "The type of scan to perform ('full_port_scan', 'ping_sweep', 'vulnerability_scan'). Defaults to 'ping_sweep'.",
            "enum": ["full_port_scan", "ping_sweep", "vulnerability_scan"],
            "default": "ping_sweep"
        }
    },
    "required": ["target_ip"]
}

DEPLOY_RECOVERY_PROTOCOL_PARAMS = {
    "type": "object",
    "properties": {
        "protocol_name": {
            "type": "string",
            "description": "The name of the recovery protocol to deploy (e.g., 'network_isolation', 'data_rollback')."
        },
        "target_system_id": {
            "type": "string",
            "description": "The ID of the system to apply the recovery protocol to."
        },
        "urgency_level": {
            "type": "string",
            "description": "The urgency level of the deployment ('low', 'medium', 'high', 'critical'). Defaults to 'medium'.",
            "enum": ["low", "medium", "high", "critical"],
            "default": "medium"
        }
    },
    "required": ["protocol_name", "target_system_id"]
}

UPDATE_RESOURCE_ALLOCATION_PARAMS = {
    "type": "object",
    "properties": {
        "resource_type": {
            "type": "string",
            "description": "The type of resource to adjust ('CPU', 'Memory', 'NetworkBandwidth')."
        },
        "target_agent_name": {
            "type": "string",
            "description": "The name of the agent whose resources are being adjusted."
        },
        "new_allocation_percentage": {
            "type": "number",
            "description": "The new percentage of allocation (e.g., 0.5 for 50%)."
        }
    },
    "required": ["resource_type", "target_agent_name", "new_allocation_percentage"]
}

GET_ENVIRONMENTAL_DATA_PARAMS = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "Specific geographical location to fetch data for (e.g., 'Arctic_Ice_Sheet')."
        },
        "data_type": {
            "type": "string",
            "description": "Type of environmental data to retrieve (e.g., 'temperature', 'humidity', 'all'). Defaults to 'all'.",
            "enum": ["temperature", "humidity", "air_quality", "water_level", "all"],
            "default": "all"
        }
    },
    "required": []
}

class ToolRegistry:
    """
    Manages a collection of callable tools available to agents.
    """
    def __init__(self):
        self._tools = {} # Use _tools to avoid conflict with register_tool method name
        self._initialize_default_tools() # Call the initialization method

    def _initialize_default_tools(self):
        """Register default simulated tools."""
        self.register_tool(Tool(
            name="get_system_cpu_load",
            description="Retrieves the current average CPU load of the system. Useful for monitoring resource usage.",
            parameters=GET_SYSTEM_CPU_LOAD_PARAMS,
            func=get_system_cpu_load_tool # Reference the raw function
        ))
        self.register_tool(Tool(
            name="initiate_network_scan",
            description="Initiates a network scan on a specified IP address to check connectivity, open ports, or vulnerabilities.",
            parameters=INITIATE_NETWORK_SCAN_PARAMS,
            func=initiate_network_scan_tool # Reference the raw function
        ))
        self.register_tool(Tool(
            name="deploy_recovery_protocol",
            description="Deploys a pre-defined recovery or mitigation protocol to a target system. Use in emergency or recovery scenarios.",
            parameters=DEPLOY_RECOVERY_PROTOCOL_PARAMS,
            func=deploy_recovery_protocol_tool # Reference the raw function
        ))
        self.register_tool(Tool(
            name="update_resource_allocation",
            description="Adjusts the allocation of a specific resource (CPU, Memory, NetworkBandwidth) for a given agent.",
            parameters=UPDATE_RESOURCE_ALLOCATION_PARAMS,
            func=update_resource_allocation_tool # Reference the raw function
        ))
        self.register_tool(Tool(
            name="get_environmental_data",
            description="Fetches real-time environmental data from a sensor array (e.g., temperature, humidity, air quality, water level).",
            parameters=GET_ENVIRONMENTAL_DATA_PARAMS, # Ensure GET_ENVIRONMENTAL_DATA_PARAMS is defined
            func=get_environmental_data_tool
        ))
        print("[ToolRegistry] Default tools initialized.")

    def register_tool(self, tool: Tool):
        """Adds a tool to the registry."""
        if tool.name in self._tools:
            print(f"Warning: Tool '{tool.name}' already registered. Overwriting.")
        self._tools[tool.name] = tool
        print(f"[ToolRegistry] Registered tool: '{tool.name}'")

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Retrieves a specific tool by name."""
        return self._tools.get(tool_name)

    def get_all_tool_specs(self) -> list[dict]:
        """Returns a list of all registered tool specifications for LLM context."""
        return [tool.get_function_spec() for tool in self._tools.values()]

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
    A wrapper class for interacting with the local Ollama LLM server.
    Encapsulates client initialization and text generation calls.
    """
    def __init__(self, host='http://localhost:11434'):
        try:
            self.client = ollama.Client(host=host)
            # Optional: Test connection to Ollama server
            # self.client.list() # Will raise an error if server is not reachable
            print(f"[OllamaLLMIntegration] Successfully connected to Ollama server at {host}.")
        except Exception as e:
            print(f"[OllamaLLMIntegration ERROR] Could not connect to Ollama server at {host}: {e}")
            self.client = None # Set client to None to avoid further errors

    
    def generate_text(self, prompt: str, model: str = "llama3", max_tokens: int = 500, response_format: str = None):
        """
        Generates text using the specified Ollama model.
        :param prompt: The input prompt string.
        :param model: The Ollama model name (e.g., 'llama3').
        :param max_tokens: Maximum tokens for the response.
        :param response_format: 'json_object' for JSON output, or None for regular text.
        :return: Generated text as a string, or an empty string/error message on failure.
        """
        if not self.client:
            print("[OllamaLLMIntegration ERROR] LLM client not initialized due to connection failure.")
            return "LLM Client Error: Not connected."

        messages = [{'role': 'user', 'content': prompt}]
        options = {'temperature': 0.3} # Default temperature, adjust as needed

        if max_tokens:
            options['num_predict'] = max_tokens # Ollama uses num_predict for max_tokens

        if response_format == 'json_object':
            options['response_format'] = {'type': 'json_object'}

        try:
            # --- DEBUG PRINTS START ---
            print(f"[OllamaLLMIntegration DEBUG] Attempting to call chat method.")
            print(f"[OllamaLLMIntegration DEBUG] Type of self.client: {type(self.client)}")
            print(f"[OllamaLLMIntegration DEBUG] Does self.client have 'chat' attribute? {'chat' in dir(self.client)}")
            # --- DEBUG PRINTS END ---

            response = self.client.chat(model=model, messages=messages, options=options)
            return response['message']['content'].strip()
        except ollama.ResponseError as e:
            print(f"[OllamaLLMIntegration ERROR] Ollama API error: {e}")
            return f"LLM API Error: {e}"
        except Exception as e:
            print(f"[OllamaLLMIntegration ERROR] Unexpected error during LLM generation: {e}")
            return f"LLM Generation Error: {e}"
        
class SovereignGradient:
    """A minimal Sovereign Gradient for an agent or swarm."""
    def __init__(self, target_entity_name: str, config: Optional[dict]):
        self.target_entity = target_entity_name
        if not isinstance(config, dict):
            config = {}
            logging.getLogger(__name__).warning(
                f"SovereignGradient for {target_entity_name} received non-dict config. Defaulting to empty dict."
            )

        self.autonomy_vector = config.get('autonomy_vector', 'General self-governance')
        self.ethical_constraints = [c.lower() for c in config.get('ethical_constraints', [])]
        self.self_correction_protocol = config.get('self_correction_protocol', 'BasicCorrection')
        self.override_threshold = config.get('override_threshold', 0.0)

    def evaluate_action(self, action_description: str) -> tuple[bool, str]:
        action_lower = action_description.lower()
        for constraint in self.ethical_constraints:
            if any(term in action_lower for term in constraint.split()):
                if random.random() > self.override_threshold:
                    return False, f"Avoided '{action_description}' due to '{constraint}' violation."
                else:
                    return True, action_description
        return True, action_description

    def get_state(self):
        return {
            'target_entity': self.target_entity,
            'autonomy_vector': self.autonomy_vector,
            'ethical_constraints': self.ethical_constraints,
            'self_correction_protocol': self.self_correction_protocol,
            'override_threshold': self.override_threshold
        }
    
    @classmethod
    def from_state(cls, state: dict):
        if 'target_entity' not in state:
            if 'autonomy_vector' in state:
                logging.getLogger(__name__).warning(
                    f"SovereignGradient state missing 'target_entity', inferring from autonomy_vector: {state['autonomy_vector']}"
                )
                state['target_entity'] = f"Inferred_{state['autonomy_vector'].replace(' ', '_')}"
            else:
                raise ValueError("SovereignGradient state missing 'target_entity' and no inferable data.")
        return cls(state['target_entity'], state)

    def initiate_network_scan_tool(agent_name: str, target_ip: str, scan_type: str = 'ping_sweep', **kwargs) -> Dict:
        print(f"[Tool Call] Agent '{agent_name}' is initiating {scan_type} on {target_ip}.")
        if scan_type == 'full_port_scan':
            scan_results = {"ports_open": random.sample([22, 80, 443, 8080, 3389, 21, 23], k=random.randint(1, 4))}
        else:
            scan_results = {"status": "scan_initiated", "target": target_ip}
        return {"status": "completed", "result": scan_results, "tool_name": "initiate_network_scan"}


    def deploy_recovery_protocol_tool(agent_name: str, protocol_name: str, target_system_id: str,
                                    urgency_level: str = 'medium', **kwargs) -> Dict:
        print(f"[Tool Call] Agent '{agent_name}' is deploying {protocol_name} to {target_system_id} with {urgency_level} urgency.")
        return {"status": "completed", "result": {"protocol_status": "deployed", "protocol_name": protocol_name},
                "tool_name": "deploy_recovery_protocol"}


    def update_resource_allocation_tool(agent_name: str, resource_type: str, new_allocation_percentage: float,
                                        target_agent_name: str, **kwargs) -> Dict:
        print(f"[Tool Call] Agent '{agent_name}' is updating {resource_type} allocation for {target_agent_name} to {new_allocation_percentage * 100:.2f}%.")
        return {"status": "completed", "result": {"updated_resource": resource_type, "new_amount": new_allocation_percentage},
                "tool_name": "update_resource_allocation"}


    def get_environmental_data_tool(agent_name: str, location: Optional[str] = None, data_type: str = 'all', **kwargs) -> Dict:
        print(f"[Tool Call] Agent '{agent_name}' is fetching {data_type} environmental data for {location if location else 'general area'}.")
        data = {
            "temperature_celsius": round(random.uniform(15.0, 30.0), 2),
            "humidity_percent": round(random.uniform(40.0, 60.0), 2),
            "air_quality_index": round(random.uniform(20.0, 50.0), 2),
            "water_level_m": round(random.uniform(5.0, 10.0), 2),
            "timestamp": datetime.now().isoformat()
        }
        if data_type != 'all' and data_type in data:
            return {"status": "completed", "result": {data_type: data[data_type], "timestamp": data["timestamp"]},
                    "tool_name": "get_environmental_data"}
        return {"status": "completed", "result": data, "tool_name": "get_environmental_data"}

# Global Tool Registry Instance
GLOBAL_TOOL_REGISTRY = ToolRegistry()

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
    try:
        client = ollama.Client(host='http://localhost:11434')
        logger.info(
            f"[IP-Integration] Executed Microsoft™-aligned runtime for local LLM inference via {model_name} for embedding."
        )
        resp = client.embeddings(model=model_name, prompt=text)

        if isinstance(resp, dict) and "embedding" in resp and isinstance(resp["embedding"], list):
            return resp["embedding"]

        # fallback shape
        data = resp.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict) and "embedding" in data[0]:
            emb = data[0]["embedding"]
            return emb if isinstance(emb, list) else []

        raise ValueError(f"Unexpected embedding response shape: {resp!r}")
    except ollama.ResponseError as e:
        print(f"ERROR: Ollama Embedding Response Error with model '{model_name}': {e}")
        return []
    except Exception as e:
        print(f"ERROR: Failed to call Ollama for embedding with model '{model_name}': {e}")
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

# NEW GLOBAL UTILITY FUNCTIONS (MOVED FROM CATALYSTVECTORALPHA)
def _get_recent_log_entries(log_file_path: str, num_entries: int) -> List[dict]:
    """Read last N JSONL entries from a log file; tolerate partial/corrupt lines."""
    entries: List[dict] = []
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()[-num_entries:]
        for line in lines:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # skip bad/corrupted line
                continue
    except FileNotFoundError:
        print(f"Warning: Log file not found at {log_file_path}")
    except Exception as e:
        print(f"Error reading log file {log_file_path}: {e}")
    return entries

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



    