#!/usr/bin/env python3
"""
CVA Memory Inspector - Always check the RIGHT ChromaDB location
"""
import chromadb
import requests
from datetime import datetime

# CRITICAL: CVA stores in persistence_data/chroma_db (NOT .chromadb)
CVA_CHROMADB_PATH = "persistence_data/chroma_db"

def get_ollama_embedding(text, model="mxbai-embed-large"):
    """Get embedding using CVA's embedding model"""
    resp = requests.post('http://localhost:11434/api/embeddings',
                        json={'model': model, 'prompt': text})
    return resp.json()['embedding']

def check_memory_status():
    """Show CVA's current memory state"""
    print("="*60)
    print("CVA MEMORY STATUS")
    print("="*60)
    print(f"ChromaDB Path: {CVA_CHROMADB_PATH}\n")
    
    client = chromadb.PersistentClient(path=CVA_CHROMADB_PATH)
    cols = client.list_collections()
    
    print(f"Total Collections: {len(cols)}")
    print(f"Timestamp: {datetime.now()}\n")
    
    # Core agents
    core_agents = ['planner', 'observer', 'worker', 'security', 'notifier']
    print("CORE AGENTS:")
    for agent in core_agents:
        col_name = f"agent-protoagent-{agent}-instance-1-memories"
        try:
            col = client.get_collection(col_name)
            print(f"  {agent.capitalize():12} {col.count():5} memories")
        except:
            print(f"  {agent.capitalize():12}     0 memories (not found)")
    
    # Dynamic agents
    print("\nDYNAMIC AGENTS:")
    dynamic = [c for c in cols if 'protoagent' not in c.name.lower() and c.count() > 0]
    for col in sorted(dynamic, key=lambda x: x.count(), reverse=True)[:10]:
        name = col.name.replace('agent-', '').replace('-memories', '')
        print(f"  {name:30} {col.count():5} memories")
    
    if len(dynamic) > 10:
        print(f"  ... and {len(dynamic)-10} more agents")
    
    print("\n" + "="*60)
    return client

def query_memory(agent_name, query_text, n=3):
    """Query a specific agent's memory"""
    client = chromadb.PersistentClient(path=CVA_CHROMADB_PATH)
    col_name = f"agent-protoagent-{agent_name}-instance-1-memories"
    
    try:
        col = client.get_collection(col_name)
        query_emb = get_ollama_embedding(query_text)
        results = col.query(query_embeddings=[query_emb], n_results=n)
        
        print(f"\n🔍 {agent_name.upper()}'s memories about: '{query_text}'")
        print("-"*60)
        for doc in results['documents'][0]:
            print(doc[:250] + "...\n")
        
    except Exception as e:
        print(f"Error querying {agent_name}: {e}")

if __name__ == "__main__":
    import sys
    
    # Check overall status
    check_memory_status()
    
    # Query if arguments provided
    if len(sys.argv) >= 3:
        agent = sys.argv[1]  # planner, observer, worker, security
        query = " ".join(sys.argv[2:])
        query_memory(agent, query)
    else:
        print("\nUsage for queries:")
        print("  python3 check_cva_memory.py <agent> <query>")
        print("  Example: python3 check_cva_memory.py planner CPU scaling decisions")
