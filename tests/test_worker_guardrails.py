"""
Test Worker guardrails - verify incomplete tool calls are blocked
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from catalyst_vector_alpha import CatalystVectorAlpha
from shared_models import MessageBus, EventMonitor
from tool_registry import tool_registry
import logging
import json
import time

# Setup logging
logger = logging.getLogger("test_guardrails")
logger.setLevel(logging.INFO)

def initialize_cva_for_testing():
    """Initialize CVA the same way app.py does"""
    
    # Create shared infrastructure
    message_bus_instance = MessageBus()
    event_monitor_instance = EventMonitor()
    
    # Create CVA instance
    system_instance = CatalystVectorAlpha(
        message_bus=message_bus_instance,
        tool_registry=tool_registry,
        event_monitor=event_monitor_instance,
        external_log_sink=logger
    )
    
    # Give it a moment to initialize agents
    time.sleep(2)
    
    return system_instance


def get_worker_agent(cva):
    """Get Worker agent from CVA agent_instances"""
    for key, agent in cva.agent_instances.items():
        if "Worker" in key or agent.name == "Worker":
            return agent
    return None


def test_missing_required_args():
    """Test that Worker blocks k8s_scale_deployment without required args"""
    
    print("\n=== Initializing CVA ===")
    cva = initialize_cva_for_testing()
    
    # Get Worker reference
    worker = get_worker_agent(cva)
    assert worker is not None, "Worker agent not found"
    print(f"Found Worker: {worker.name}")
    
    # Missing 'namespace' and 'replicas' - should be BLOCKED
    broken_task = {
        "task_id": "test_guardrail_1",
        "tool_name": "k8s_scale_deployment",
        "arguments": {
            "deployment_name": "nginx"
            # Missing: namespace, replicas
        },
        "context": "Testing guardrails"
    }
    
    print("\n=== TEST 1: Missing required args ===")
    print(f"Sending: {json.dumps(broken_task, indent=2)}")
    
    result = worker.execute_tool_call(broken_task)
    
    print(f"\nResult status: {result.get('status')}")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    assert result['status'] == 'INVALID_ARGS', f"Expected INVALID_ARGS, got {result['status']}"
    print("✓ Worker correctly blocked execution")
    
    cva.shutdown()
    return True


def test_partial_args():
    """Test missing only one required arg"""
    
    print("\n=== Initializing CVA ===")
    cva = initialize_cva_for_testing()
    
    worker = get_worker_agent(cva)
    assert worker is not None, "Worker agent not found"
    
    partial_task = {
        "task_id": "test_guardrail_2",
        "tool_name": "k8s_scale_deployment",
        "arguments": {
            "deployment_name": "nginx",
            "namespace": "default"
            # Missing: replicas
        },
        "context": "Testing partial args"
    }
    
    print("\n=== TEST 2: Partial args (missing replicas) ===")
    print(f"Sending: {json.dumps(partial_task, indent=2)}")
    
    result = worker.execute_tool_call(partial_task)
    
    print(f"\nResult status: {result.get('status')}")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    assert result['status'] == 'INVALID_ARGS', f"Expected INVALID_ARGS, got {result['status']}"
    print("✓ Worker correctly blocked execution")
    
    cva.shutdown()
    return True


def test_valid_args_structure():
    """Test that valid args pass guardrails"""
    
    print("\n=== Initializing CVA ===")
    cva = initialize_cva_for_testing()
    
    worker = get_worker_agent(cva)
    assert worker is not None, "Worker agent not found"
    
    valid_task = {
        "task_id": "test_guardrail_3",
        "tool_name": "k8s_scale_deployment",
        "arguments": {
            "deployment_name": "nginx",
            "namespace": "default",
            "replicas": 3
        },
        "context": "Testing valid args"
    }
    
    print("\n=== TEST 3: Valid args structure ===")
    print(f"Sending: {json.dumps(valid_task, indent=2)}")
    
    result = worker.execute_tool_call(valid_task)
    
    print(f"\nResult status: {result.get('status')}")
    
    # Should NOT be INVALID_ARGS (might be EXECUTION_ERROR if k8s not running)
    assert result['status'] != 'INVALID_ARGS', f"Valid args should pass validation, got {result['status']}"
    print("✓ Valid args passed guardrails")
    
    cva.shutdown()
    return True


if __name__ == "__main__":
    print("Testing Worker Guardrails")
    print("=" * 50)
    
    try:
        test_missing_required_args()
        test_partial_args()
        test_valid_args_structure()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✓")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
