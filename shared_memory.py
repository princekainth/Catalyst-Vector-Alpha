import chromadb
from chromadb.utils import embedding_functions
import uuid
import time
import json
from datetime import datetime
import os

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

        # Ensure the directory exists
        os.makedirs(persist_path, exist_ok=True)
        
        print(f"🧠 [SharedMemory] Initializing Hive Mind at {persist_path}...")
        
        # Initialize Persistent Vector DB
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # Use standard high-performance embedding model
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or Get the Universal Collection
        self.collection = self.client.get_or_create_collection(
            name="collective_memory",
            embedding_function=self.ef
        )
        print(f"🧠 [SharedMemory] Connected. Total memories: {self.collection.count()}")
        self._initialized = True

    def add_memory(self, agent_name, text, category, metadata=None):
        """
        Records an event into the universal timeline.
        Categories: 'observation', 'plan', 'action', 'outcome', 'reflection'
        """
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
        self.collection.add(
            documents=[text],
            metadatas=[meta],
            ids=[f"{agent_name}_{str(uuid.uuid4())[:8]}"]
        )
        print(f"📝 [Memory] Recorded by {agent_name}: {text[:50]}...")

    def query_memory(self, query_text, n_results=5, agent_filter=None):
        """
        Semantic Search across the Hive Mind.
        """
        where_clause = None
        if agent_filter:
            where_clause = {"agent": agent_filter}

        results = self.collection.query(
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

# --- TEST HARNESS ---
if __name__ == "__main__":
    print("\n--- 🧪 TESTING SHARED MEMORY ---")
    memory = SharedMemory()
    memory.add_memory("Observer", "Namespace is 'default', Max Replicas is 5", "observation")
    recall = memory.query_memory("What is the namespace?")
    print(f"Recall Result: {recall[0]['text']}")
