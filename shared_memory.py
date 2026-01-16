import chromadb
from chromadb.utils import embedding_functions
import uuid
import time
import json
from datetime import datetime
import os
import logging

class SharedMemory:
    """
    The Collective Unconscious of the CVA System.
    Stores observations, decisions, and outcomes from ALL agents
    in a single semantic vector space.
    """
    _instance = None
    _lock = None

    @classmethod
    def _get_lock(cls):
        if cls._lock is None:
            from threading import Lock
            cls._lock = Lock()
        return cls._lock

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._get_lock():
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, persist_path="./persistence_data/cva_brain"):
        # Skip if already initialized (singleton)
        if getattr(self, '_initialized', False):
            return

        logger = logging.getLogger("CatalystLogger")

        # Ensure the directory exists
        os.makedirs(persist_path, exist_ok=True)
        
        logger.info(f"🧠 [SharedMemory] Initializing Hive Mind at {persist_path}...")
        self._persist_path = persist_path
        self._memory_enabled = True
        self._memory_ready = False
        self.client = None
        self.collection = None
        self.curiosity_collection = None
        self.ef = None

        if os.getenv("CVA_LAZY_MEMORY_INIT", "0") == "1":
            self._initialize_memory()
        self._initialized = True

    def _initialize_memory(self) -> None:
        if self._memory_ready or not self._memory_enabled:
            return
        logger = logging.getLogger("CatalystLogger")
        if os.getenv("CVA_DISABLE_EMBEDDINGS", "0") != "0":
            logger.info("[SharedMemory] Embeddings disabled by CVA_DISABLE_EMBEDDINGS.")
            self._memory_enabled = False
            return
        try:
            self.client = chromadb.PersistentClient(path=self._persist_path)
        except Exception as e:
            logger.warning(f"[SharedMemory] Disabled (client init failed): {e}")
            self._memory_enabled = False
            return

        try:
            offline = False
            if os.getenv("TRANSFORMERS_OFFLINE", "0").lower() not in ("", "0", "false", "no"):
                offline = True
            if os.getenv("HF_HUB_OFFLINE", "0").lower() not in ("", "0", "false", "no"):
                offline = True
            model_name = "all-MiniLM-L6-v2"
            if offline and not self._model_cached(model_name):
                logger.warning("[SharedMemory] Offline and model cache missing; memory disabled.")
                self._memory_enabled = False
                return
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        except Exception as e:
            logger.warning(f"[SharedMemory] Embeddings unavailable; memory disabled: {e}")
            self._memory_enabled = False
            return

        try:
            self.collection = self.client.get_or_create_collection(
                name="collective_memory",
                embedding_function=self.ef
            )
            self.curiosity_collection = self.client.get_or_create_collection(
                name="curiosity_memory",
                embedding_function=self.ef
            )
            self._memory_ready = True
            logger.info(f"🧠 [SharedMemory] Connected. Total memories: {self.collection.count()}")
        except Exception as e:
            logger.warning(f"[SharedMemory] Disabled (collection init failed): {e}")
            self.collection = None
            self.curiosity_collection = None
            self._memory_enabled = False

    def _ensure_ready(self) -> bool:
        if not self._memory_enabled:
            return False
        if not self._memory_ready:
            self._initialize_memory()
        return bool(self._memory_ready)

    def _add_to_collection(self, collection, agent_name, text, category, metadata=None):
        logger = logging.getLogger("CatalystLogger")
        if not self._ensure_ready() or not collection:
            logger.info("[SharedMemory] Memory disabled; skipping write.")
            return
        if metadata is None:
            metadata = {}

        # Enrich metadata
        meta = metadata.copy()
        meta.update({
            "agent": agent_name,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time()
        })

        # Add to Vector DB
        collection.add(
            documents=[text],
            metadatas=[meta],
            ids=[f"{agent_name}_{str(uuid.uuid4())[:8]}"]
        )
        logger.info(f"📝 [Memory] Recorded by {agent_name}: {text[:50]}...")

    def add_memory(self, agent_name, text, category, metadata=None):
        """
        Records an event into the universal timeline.
        Categories: 'observation', 'plan', 'action', 'outcome', 'reflection'
        """
        self._add_to_collection(self.collection, agent_name, text, category, metadata)

    def add_learning_memory(self, agent_name, text, category="learning", metadata=None):
        """Record a learning-only memory to the curiosity collection."""
        self._add_to_collection(self.curiosity_collection, agent_name, text, category, metadata)

    def query_memory(self, query_text, n_results=5, agent_filter=None, collection_name="collective"):
        """
        Semantic Search across the Hive Mind.
        """
        if not self._ensure_ready():
            return []
        if not self.collection and collection_name == "collective":
            return []
        if not self.curiosity_collection and collection_name != "collective":
            return []
        where_clause = None
        if agent_filter:
            where_clause = {"agent": agent_filter}

        target_collection = self.collection if collection_name == "collective" else self.curiosity_collection
        results = target_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_clause
        )

        # Format results for easier reading
        memories = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                memories.append({
                    "text": doc,
                    "agent": meta.get("agent"),
                    "category": meta.get("category"),
                    "timestamp": meta.get("timestamp")
                })
        return memories

    def query_learning(self, query_text, n_results=5, agent_filter=None):
        """Semantic search across curiosity-only memory."""
        return self.query_memory(query_text, n_results=n_results, agent_filter=agent_filter, collection_name="curiosity")

    def _model_cached(self, model_name: str) -> bool:
        cache_dirs = []
        hf_hub_cache = os.getenv("HF_HUB_CACHE")
        transformers_cache = os.getenv("TRANSFORMERS_CACHE")
        hf_home = os.getenv("HF_HOME")
        if hf_hub_cache:
            cache_dirs.append(hf_hub_cache)
        if transformers_cache:
            cache_dirs.append(transformers_cache)
        if hf_home:
            cache_dirs.append(os.path.join(hf_home, "hub"))
        cache_dirs.append(os.path.expanduser("~/.cache/huggingface/hub"))
        hub_dir = f"models--sentence-transformers--{model_name}"
        st_dir = os.path.join("sentence-transformers", model_name)
        for base in cache_dirs:
            if not base:
                continue
            if os.path.isdir(os.path.join(base, hub_dir)):
                return True
            if os.path.isdir(os.path.join(base, st_dir)):
                return True
        return False

# --- TEST HARNESS ---
if __name__ == "__main__":
    print("\n--- 🧪 TESTING SHARED MEMORY ---")
    memory = SharedMemory()
    recall = memory.query_memory("What is the namespace?")
    if recall:
        print(f"Recall Result: {recall[0]['text']}")
    else:
        print("Recall Result: No matches found (Memory might be initializing or empty).")
