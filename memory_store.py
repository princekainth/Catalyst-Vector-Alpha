import chromadb
from chromadb.config import Settings
import uuid
import time
import json
from collections import deque

class MemoryStore:
    def __init__(self, persist_path="persistence_data/chroma_db"):
        # 1. Setup New Brain (ChromaDB)
        try:
            self.client = chromadb.PersistentClient(path=persist_path)
            self.episodic = self.client.get_or_create_collection(name="episodic_memory")
            self.semantic = self.client.get_or_create_collection(name="semantic_knowledge")
            self.procedural = self.client.get_or_create_collection(name="code_snippets")
        except Exception as e:
            print(f"[MemoryStore] Warning: ChromaDB init failed: {e}")
            self.client = None

        # 2. Setup Old Brain (In-Memory Cache for Legacy Agents)
        self._recent_cache = deque(maxlen=1000)

    def add(self, doc_type, content, metadata=None):
        """
        Universal Add: Handles both legacy dicts and new string memories.
        """
        if metadata is None:
            metadata = {}
        
        # --- Path A: Support Legacy Agents (In-Memory) ---
        # Create a dictionary entry compatible with what agents.py expects
        legacy_entry = {
            "type": doc_type,
            "content": content,
            "timestamp": time.time()
        }
        # Merge metadata if content is a dict (legacy behavior)
        if isinstance(content, dict):
            legacy_entry.update(content)
        
        self._recent_cache.append(legacy_entry)

        # --- Path B: Support New Brain (ChromaDB) ---
        if self.client:
            # Convert dict content to string for vector storage
            text_content = str(content) if not isinstance(content, str) else content
            
            # Map types to collections
            target_col = self.episodic
            if doc_type == "semantic": target_col = self.semantic
            elif doc_type == "procedural": target_col = self.procedural
            
            try:
                target_col.add(
                    documents=[text_content],
                    metadatas=[{"type": str(doc_type), "timestamp": str(time.time())}],
                    ids=[str(uuid.uuid4())]
                )
            except Exception as e:
                print(f"[MemoryStore] Vector save failed: {e}")

        return "Memory Stored."

    # LEGACY COMPATIBILITY: 'recent' matches agents.py signature
    def recent(self, type_filter=None, limit=10):
        """
        Returns a list of recent memories, filtered by type.
        Fixes: TypeError: 'collections.deque' object is not callable
        """
        # Convert deque to list and reverse (newest first)
        all_items = list(self._recent_cache)[::-1]
        
        if type_filter:
            filtered = [item for item in all_items if item.get("type") == type_filter]
            return filtered[:limit]
        
        return all_items[:limit]

    # Tool Alias
    def add_memory(self, category, content):
        return self.add(category, content)

    def query_memory(self, category, query, n_results=3):
        if not self.client: return "Memory Offline."
        
        target_col = getattr(self, category, self.episodic)
        results = target_col.query(query_texts=[query], n_results=n_results)
        return results['documents'][0] if results['documents'] else ["No matches found."]

mem_store = MemoryStore()
