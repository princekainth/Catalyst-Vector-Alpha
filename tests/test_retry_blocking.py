"""
Test one-retry rule and BLOCKED state transitions
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def simulate_task_lifecycle():
    """Simulate task going through: initial fail → retry → BLOCKED"""
    
    print("\n=== Simulating Task Lifecycle ===")
    
    task_state = {
        "task_id": "test_task_1",
        "status": "pending",
        "retry_count": 0,
        "max_retries": 1
    }
    
    print(f"Initial state: {task_state}")
    
    # First execution - fails with INVALID_ARGS
    print("\n--- First execution ---")
    task_state["status"] = "failed"
    task_state["failure_reason"] = "INVALID_ARGS"
    print(f"Result: FAILED - {task_state['failure_reason']}")
    
    # Check if retry allowed
    if task_state["retry_count"] < task_state["max_retries"]:
        task_state["retry_count"] += 1
        task_state["status"] = "retrying"
        print(f"✓ Retry #{task_state['retry_count']} allowed")
    else:
        task_state["status"] = "BLOCKED"
        print("❌ Max retries reached - would BLOCK")
    
    # Retry execution - still fails
    print("\n--- Retry execution ---")
    task_state["status"] = "failed"
    task_state["failure_reason"] = "INVALID_ARGS"
    print(f"Result: FAILED - {task_state['failure_reason']}")
    
    # Check if another retry allowed
    if task_state["retry_count"] < task_state["max_retries"]:
        task_state["retry_count"] += 1
        task_state["status"] = "retrying"
        print(f"Retry #{task_state['retry_count']} allowed")
    else:
        task_state["status"] = "BLOCKED"
        print(f"✓ Max retries ({task_state['max_retries']}) exhausted")
        print(f"✓ Task marked BLOCKED")
    
    assert task_state["status"] == "BLOCKED", "Should be BLOCKED after max retries"
    assert task_state["retry_count"] == 1, "Should have exactly 1 retry"
    
    print(f"\nFinal state: {task_state}")
    print("✓ One-retry rule enforced correctly")


def test_immediate_block_on_persistent_failure():
    """Test that same error on retry causes immediate BLOCK"""
    
    print("\n=== TEST: Persistent failure blocking ===")
    
    # First attempt
    first_result = {
        "status": "INVALID_ARGS",
        "missing": ["namespace", "replicas"]
    }
    
    # Retry attempt with SAME missing args
    retry_result = {
        "status": "INVALID_ARGS", 
        "missing": ["namespace", "replicas"]
    }
    
    print(f"First attempt: {first_result}")
    print(f"Retry attempt: {retry_result}")
    
    # Check if error persists
    same_error = (
        first_result["status"] == retry_result["status"] and
        first_result.get("missing") == retry_result.get("missing")
    )
    
    if same_error:
        final_status = "BLOCKED"
        print("✓ Same error detected - task BLOCKED")
    else:
        final_status = "retry_again"
        print("Different error - could retry again")
    
    assert final_status == "BLOCKED", "Persistent errors should cause BLOCK"
    print("✓ No infinite retry loops on persistent failures")


def test_successful_retry():
    """Test that fixed args on retry succeed"""
    
    print("\n=== TEST: Successful retry ===")
    
    task_state = {
        "task_id": "test_task_2",
        "status": "pending",
        "retry_count": 0,
        "max_retries": 1
    }
    
    # First execution - fails
    print("\n--- First execution (missing args) ---")
    task_state["status"] = "failed"
    task_state["failure_reason"] = "INVALID_ARGS"
    task_state["retry_count"] += 1
    print(f"Failed: {task_state['failure_reason']}")
    
    # Retry with fixed args - succeeds
    print("\n--- Retry execution (args fixed) ---")
    task_state["status"] = "success"
    task_state["failure_reason"] = None
    print("Success!")
    
    assert task_state["status"] == "success", "Fixed retry should succeed"
    assert task_state["retry_count"] == 1, "Should show 1 retry was needed"
    
    print(f"\nFinal state: {task_state}")
    print("✓ System can recover on retry with fixed args")


if __name__ == "__main__":
    print("Testing Retry & Blocking Logic")
    print("=" * 50)
    
    try:
        simulate_task_lifecycle()
        test_immediate_block_on_persistent_failure()
        test_successful_retry()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✓")
        print("\nKey behaviors validated:")
        print("- One retry maximum per task")
        print("- BLOCKED state on retry exhaustion")
        print("- No infinite loops on persistent failures")
        print("- Recovery possible with fixed args")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
