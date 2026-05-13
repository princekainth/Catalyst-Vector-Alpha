
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add current dir to path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry

def test_safe_call():
    registry = ToolRegistry()
    print("--- Testing ToolRegistry.safe_call() ---")
    
    # safe_call should work without bypass
    result = registry.safe_call("get_pod_status", agent_id="TestAgent", namespace="default")
    print(f"Result Status: {result.get('status')}")
    if result.get('status') == 'ok':
        print("SUCCESS: ToolRegistry.safe_call() executed correctly")
    else:
        print(f"FAILED: ToolRegistry.safe_call() returned error: {result.get('error')}")

if __name__ == "__main__":
    test_safe_call()
