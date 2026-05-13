import time
from evolution_agent import EvolutionAgent
from database import cva_db

def verify_persistence():
    print("--- Starting Evolution Persistence Verification ---")
    
    # 1. Clear existing state
    print("[1] Clearing existing evolution state...")
    cva_db.save_evolution_state([], [], [])
    
    # 2. Create agent and add a gap
    print("[2] Creating Agent 1 and recording gap...")
    agent1 = EvolutionAgent(db=cva_db)
    agent1.record_capability_gap(
        description="Verification Gap",
        context="Test context",
        source_agent="PersistenceVerifier"
    )
    
    gap_id = agent1.capability_gaps[0]["id"]
    print(f"Recorded gap: {gap_id}")
    
    # 3. Destroy agent 1
    del agent1
    
    # 4. Create agent 2 and verify restoration
    print("[3] Creating Agent 2 (simulated restart)...")
    agent2 = EvolutionAgent(db=cva_db)
    
    print(f"Agent 2 Gaps: {len(agent2.capability_gaps)}")
    if len(agent2.capability_gaps) > 0 and agent2.capability_gaps[0]["id"] == gap_id:
        print("✅ SUCCESS: Capability gap persisted and restored!")
    else:
        print("❌ FAILURE: Capability gap lost or mismatched.")
        exit(1)

    # 5. Add a history item and verify again
    print("[4] Adding history item and verifying restart...")
    agent2.evolution_history.append({
        "tool_name": "test_tool",
        "status": "quarantined",
        "timestamp": "now"
    })
    agent2._persist_state()
    
    del agent2
    agent3 = EvolutionAgent(db=cva_db)
    
    if len(agent3.evolution_history) > 0 and agent3.evolution_history[0]["tool_name"] == "test_tool":
        print("✅ SUCCESS: Evolution history persisted and restored!")
    else:
        print("❌ FAILURE: Evolution history lost.")
        exit(1)

    print("\n--- Persistence Verification Complete: PASSED ---")

if __name__ == "__main__":
    verify_persistence()
