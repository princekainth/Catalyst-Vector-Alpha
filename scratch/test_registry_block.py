
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add current dir to path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry

def test_block():
    registry = ToolRegistry()
    print("--- Testing ToolRegistry.call() Block ---")
    
    # Ensure tool exists for the test
    try:
        registry.call("get_pod_status", namespace="default")
        print("FAILED: ToolRegistry.call() did not raise RuntimeError")
    except RuntimeError as e:
        print(f"SUCCESS: Caught expected RuntimeError: {e}")
    except Exception as e:
        print(f"FAILED: Caught unexpected exception type: {type(e).__name__}: {e}")

def test_bypass():
    registry = ToolRegistry()
    print("\n--- Testing ToolRegistry.call() Bypass ---")
    os.environ["CVA_ALLOW_UNSAFE_TOOL_CALL"] = "1"
    
    try:
        registry.call("get_pod_status", namespace="default")
        print("SUCCESS: ToolRegistry.call() allowed with bypass env var")
    except Exception as e:
        print(f"FAILED: ToolRegistry.call() raised error despite bypass: {e}")

if __name__ == "__main__":
    test_block()
    test_bypass()
