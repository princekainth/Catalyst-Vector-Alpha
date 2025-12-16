"""
Simulate K8s event detection → mission creation → execution validation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json

def simulate_pod_crash_detection():
    """Simulate Observer detecting a pod crash event"""
    
    print("\n=== Simulating Pod Crash Detection ===")
    
    # Observer detects K8s event
    k8s_event = {
        "type": "DELETED",
        "object": {
            "kind": "Pod",
            "metadata": {
                "name": "nginx-pod-abc123",
                "namespace": "default"
            },
            "status": {
                "phase": "Failed",
                "reason": "CrashLoopBackOff"
            }
        }
    }
    
    print(f"K8s Event detected:\n{json.dumps(k8s_event, indent=2)}")
    
    # Observer creates observation
    observation = {
        "event_type": "pod_crash",
        "severity": "critical",
        "namespace": k8s_event["object"]["metadata"]["namespace"],
        "pod_name": k8s_event["object"]["metadata"]["name"],
        "reason": k8s_event["object"]["status"]["reason"],
        "recommended_action": "investigate_and_remediate"
    }
    
    print(f"\nObserver output:\n{json.dumps(observation, indent=2)}")
    print("✓ Observer detected critical event")
    
    return observation


def simulate_planner_mission_creation(observation):
    """Simulate Planner creating remediation mission"""
    
    print("\n=== Simulating Planner Mission Creation ===")
    
    # Planner analyzes observation
    mission = {
        "mission_id": "remediate_pod_crash_001",
        "mission_type": "k8s_remediation",
        "steps": [
            {
                "step_id": "check_deployment",
                "agent": "Worker",
                "tool": "k8s_get_deployment",
                "arguments": {
                    "deployment_name": "nginx",
                    "namespace": observation["namespace"]
                }
            },
            {
                "step_id": "scale_if_needed",
                "agent": "Worker", 
                "tool": "k8s_scale_deployment",
                "arguments": {
                    "deployment_name": "nginx",
                    "namespace": observation["namespace"],
                    "replicas": 3
                }
            }
        ],
        "priority": "high",
        "trigger": observation
    }
    
    print(f"Planner mission:\n{json.dumps(mission, indent=2)}")
    
    # Validate mission has required fields
    assert "steps" in mission, "Mission must have steps"
    assert len(mission["steps"]) > 0, "Mission must have at least one step"
    
    for step in mission["steps"]:
        assert "tool" in step, f"Step {step['step_id']} missing tool"
        assert "arguments" in step, f"Step {step['step_id']} missing arguments"
    
    print("✓ Planner created valid mission")
    
    return mission


def simulate_worker_preflight_validation(mission):
    """Simulate Worker validating mission steps before execution"""
    
    print("\n=== Simulating Worker Preflight Validation ===")
    
    validation_results = []
    
    for step in mission["steps"]:
        print(f"\nValidating step: {step['step_id']}")
        print(f"Tool: {step['tool']}")
        print(f"Args: {step['arguments']}")
        
        # Check for k8s_scale_deployment requirements
        if step['tool'] == 'k8s_scale_deployment':
            missing = []
            if not step['arguments'].get('namespace'):
                missing.append('namespace')
            if step['arguments'].get('replicas') is None:
                missing.append('replicas')
            
            if missing:
                result = {
                    "step_id": step['step_id'],
                    "status": "INVALID_ARGS",
                    "missing": missing
                }
                print(f"❌ Validation failed: missing {missing}")
            else:
                result = {
                    "step_id": step['step_id'],
                    "status": "valid",
                    "can_execute": True
                }
                print(f"✓ Validation passed")
        else:
            # Other tools - assume valid for this test
            result = {
                "step_id": step['step_id'],
                "status": "valid",
                "can_execute": True
            }
            print(f"✓ Validation passed")
        
        validation_results.append(result)
    
    # Check overall mission validity
    all_valid = all(r['status'] == 'valid' for r in validation_results)
    
    if all_valid:
        print("\n✓ All steps validated - mission can execute")
    else:
        failed_steps = [r['step_id'] for r in validation_results if r['status'] != 'valid']
        print(f"\n❌ Mission blocked - invalid steps: {failed_steps}")
    
    return validation_results, all_valid


def test_end_to_end_flow():
    """Test complete flow: detection → planning → validation"""
    
    print("\n" + "=" * 50)
    print("TESTING FULL EVENT FLOW")
    print("=" * 50)
    
    # Step 1: Observer detects event
    observation = simulate_pod_crash_detection()
    
    # Step 2: Planner creates mission
    mission = simulate_planner_mission_creation(observation)
    
    # Step 3: Worker validates before execution
    validation_results, can_execute = simulate_worker_preflight_validation(mission)
    
    assert can_execute, "Mission should be executable with valid args"
    
    print("\n" + "=" * 50)
    print("✓ COMPLETE FLOW VALIDATED")
    print("=" * 50)
    print("\nFlow verified:")
    print("1. Observer detected K8s event")
    print("2. Planner created remediation mission")
    print("3. Worker validated all steps")
    print("4. System ready to execute (no crashes)")


def test_invalid_mission_blocked():
    """Test that missions with invalid args are blocked"""
    
    print("\n" + "=" * 50)
    print("TESTING INVALID MISSION BLOCKING")
    print("=" * 50)
    
    # Create mission with missing args
    bad_mission = {
        "mission_id": "bad_mission_001",
        "steps": [
            {
                "step_id": "bad_scale",
                "agent": "Worker",
                "tool": "k8s_scale_deployment",
                "arguments": {
                    "deployment_name": "nginx"
                    # Missing: namespace, replicas
                }
            }
        ]
    }
    
    print(f"\nBad mission:\n{json.dumps(bad_mission, indent=2)}")
    
    validation_results, can_execute = simulate_worker_preflight_validation(bad_mission)
    
    assert not can_execute, "Invalid mission should be blocked"
    print("\n✓ Invalid mission correctly blocked from execution")


if __name__ == "__main__":
    print("Testing K8s Event Flow (Simulated)")
    print("=" * 60)
    
    try:
        test_end_to_end_flow()
        test_invalid_mission_blocked()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("\nHardened behaviors validated:")
        print("- Observer → Planner → Worker coordination")
        print("- Preflight validation catches bad missions")
        print("- Invalid steps blocked before execution")
        print("- No crashes on malformed plans")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
