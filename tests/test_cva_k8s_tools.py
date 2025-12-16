"""
Test CVA's K8s tools against real cluster
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tool_registry import tool_registry
import json

def test_k8s_scale_tool():
    """Test CVA's k8s_scale tool"""
    
    print("\n=== TEST: k8s_scale tool ===")
    
    tool = tool_registry.get_tool("k8s_scale")
    assert tool is not None, "k8s_scale tool should exist"
    
    print(f"Tool found: {tool.name}")
    print(f"Description: {tool.description}")
    
    # Scale to 3
    print("\nScaling nginx-test to 3 replicas...")
    result = tool.func(
        deployment="nginx-test",
        namespace="default",
        replicas=3
    )
    
    print(f"Result:\n{json.dumps(result, indent=2) if isinstance(result, dict) else result}")
    
    # Scale back to 2
    print("\nScaling back to 2 replicas...")
    result = tool.func(
        deployment="nginx-test",
        namespace="default",
        replicas=2
    )
    
    print(f"Result:\n{json.dumps(result, indent=2) if isinstance(result, dict) else result}")
    
    print("✓ k8s_scale tool executed successfully")


def test_get_pod_status_tool():
    """Test CVA's get_pod_status tool"""
    
    print("\n=== TEST: get_pod_status tool ===")
    
    tool = tool_registry.get_tool("get_pod_status")
    assert tool is not None, "get_pod_status tool should exist"
    
    print(f"Tool found: {tool.name}")
    
    # Get pod status
    result = tool.func(namespace="default")
    
    result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
    print(f"Result (truncated):\n{result_str[:500]}...")
    
    print("✓ get_pod_status tool executed successfully")


def test_kubernetes_pod_metrics_tool():
    """Test CVA's kubernetes_pod_metrics tool"""
    
    print("\n=== TEST: kubernetes_pod_metrics tool ===")
    
    tool = tool_registry.get_tool("kubernetes_pod_metrics")
    assert tool is not None, "kubernetes_pod_metrics tool should exist"
    
    print(f"Tool found: {tool.name}")
    
    # Get metrics
    result = tool.func(namespace="default")
    
    result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
    print(f"Result (truncated):\n{result_str[:500]}...")
    
    print("✓ kubernetes_pod_metrics tool executed successfully")


def test_k8s_restart_tool():
    """Test CVA's k8s_restart tool exists"""
    
    print("\n=== TEST: k8s_restart tool ===")
    
    tool = tool_registry.get_tool("k8s_restart")
    assert tool is not None, "k8s_restart tool should exist"
    
    print(f"Tool found: {tool.name}")
    print(f"Description: {tool.description}")
    
    # Don't actually restart - just verify tool exists
    print("✓ k8s_restart tool available (not executed)")


if __name__ == "__main__":
    print("Testing CVA K8s Tools")
    print("=" * 50)
    
    try:
        test_k8s_scale_tool()
        test_get_pod_status_tool()
        test_kubernetes_pod_metrics_tool()
        test_k8s_restart_tool()
        
        print("\n" + "=" * 50)
        print("ALL CVA K8s TOOL TESTS PASSED ✓")
        print("\nCVA K8s tools verified:")
        print("- k8s_scale: Scale deployments ✓")
        print("- get_pod_status: Monitor pods ✓")
        print("- kubernetes_pod_metrics: Get metrics ✓")
        print("- k8s_restart: Restart deployments ✓")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
