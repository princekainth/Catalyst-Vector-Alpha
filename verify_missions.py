
import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.mission_policy import candidate_missions, MISSION_TOOL_POLICY

def verify_digital_nomad():
    cands = candidate_missions()
    print(f"Candidate Missions: {cands}")
    
    required = ["cybersecurity", "system_architect", "proactive_dev"]
    missing = [m for m in required if m not in cands]
    
    if missing:
        print(f"FAILED: Missing candidate missions: {missing}")
        return False
    
    print("SUCCESS: All new missions found in candidates.")
    
    for m in required:
        if m not in MISSION_TOOL_POLICY:
            print(f"FAILED: Mission '{m}' not found in MISSION_TOOL_POLICY")
            return False
        policy = MISSION_TOOL_POLICY[m]
        print(f"Policy for '{m}': {policy}")
        if not policy.get("allow"):
             print(f"FAILED: Mission '{m}' has no allowed tools")
             return False
             
    print("SUCCESS: All mission tool policies verified.")
    return True

if __name__ == "__main__":
    if verify_digital_nomad():
        sys.exit(0)
    else:
        sys.exit(1)
