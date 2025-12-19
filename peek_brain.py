from shared_memory import SharedMemory

# Connect to the existing brain
print("🧠 Connecting to the Hive Mind...")
brain = SharedMemory()

# Ask it what it has done recently
print("\n--- 🔍 RECENT MEMORIES ---")
# We search for "task" to see the work logs
memories = brain.query_memory("task", n_results=10)

for i, mem in enumerate(memories):
    print(f"\n[{i+1}] AGENT: {mem['agent']} ({mem['category']})")
    print(f"    TIME:  {mem['timestamp']}")
    print(f"    THOUGHT: {mem['text']}")