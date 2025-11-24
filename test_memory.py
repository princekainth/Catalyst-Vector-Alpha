from memory_store import MemoryStore

try:
    # Test creating database and adding memory
    print("Creating MemoryStore...")
    mem = MemoryStore()
    print("MemoryStore created successfully")
    
    print("Adding test memory...")
    mem.add("TestAgent", "TestMemory", {"test": "data", "value": 123})
    print("Memory added successfully")
    
    # Test retrieving
    print("Retrieving recent memories...")
    recent = mem.recent(limit=5)
    print(f"Found {len(recent)} memories")
    print(recent)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()