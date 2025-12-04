#!/usr/bin/env python3
"""
Manual SystemAlert Injection Script for Testing K8s LLM Decisions
"""

import time
import json
from datetime import datetime, timezone


def timestamp_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def inject_high_cpu_alert(cva):
    """Inject high CPU alert into Planner's memory"""
    
    alert = {
        "type": "high_cpu_load",
        "avg_cpu": 85.5,
        "target_deployment": "nginx",
        "current_replicas": 2,
        "recommended_replicas": 4,
        "timestamp": time.time()
    }
    
    # Find Planner agent
    planner = None
    for name, agent in cva.agent_instances.items():
        if hasattr(agent, 'role') and agent.role == "strategic_planner":
            planner = agent
            print(f"✓ Found Planner: {name}")
            break
    
    if not planner:
        print("✗ Planner not found")
        return False
    
    # Inject SystemAlert memory
    try:
        planner.memetic_kernel.add_memory(
            memory_type="SystemAlert",
            content=alert,
            timestamp=timestamp_now()
        )
        
        print(f"\n✓ Injected High CPU Alert:")
        print(f"  CPU: {alert['avg_cpu']}%")
        print(f"  Deployment: {alert['target_deployment']}")
        print(f"  Current Replicas: {alert['current_replicas']}")
        print("\nWatch CVA logs for K8S_LLM_DECISION...")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("="*70)
    print("  K8s Alert Injection Test")
    print("="*70)
    
    from catalyst_vector_alpha import CatalystVectorAlpha
    
    # Load CVA from state
    try:
        import os
        state_file = "persistence_data/state.json"
        
        if os.path.exists(state_file):
            print(f"✓ Loading from {state_file}")
            cva = CatalystVectorAlpha.load_state_from_file(state_file)
        else:
            print("⚠ No state file, creating new instance")
            cva = CatalystVectorAlpha()
            
    except Exception as e:
        print(f"✗ Error loading CVA: {e}")
        exit(1)
    
    # Inject alert
    inject_high_cpu_alert(cva)
    
    print("\n" + "="*70)
    print("  Next: Check CVA terminal for LLM decision logs")
    print("="*70)