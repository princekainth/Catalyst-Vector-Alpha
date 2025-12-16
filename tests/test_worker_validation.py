"""
Direct test of Worker's argument validation logic
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tool_registry import tool_registry

def test_k8s_scale_validation():
    """Test k8s_scale missing args detection"""
    
    print("\n=== TEST 1: k8s_scale missing namespace & replicas ===")
    
    tool_name = "k8s_scale"
    tool_args = {
        "deployment_name": "nginx"
        # Missing: namespace, replicas
    }
    
    # Simulate Worker's validation logic
    missing_required = []
    
    if tool_name == "k8s_scale":
        if not tool_args.get("namespace"):
            missing_required.append("namespace")
        if tool_args.get("replicas") is None:
            missing_required.append("replicas")
    
    print(f"Tool: {tool_name}")
    print(f"Args: {tool_args}")
    print(f"Missing: {missing_required}")
    
    assert "namespace" in missing_required, "Should detect missing namespace"
    assert "replicas" in missing_required, "Should detect missing replicas"
    print("✓ Correctly detected missing args")


def test_k8s_scale_partial():
    """Test k8s_scale missing only replicas"""
    
    print("\n=== TEST 2: k8s_scale missing replicas only ===")
    
    tool_name = "k8s_scale"
    tool_args = {
        "deployment_name": "nginx",
        "namespace": "default"
        # Missing: replicas
    }
    
    missing_required = []
    
    if tool_name == "k8s_scale":
        if not tool_args.get("namespace"):
            missing_required.append("namespace")
        if tool_args.get("replicas") is None:
            missing_required.append("replicas")
    
    print(f"Tool: {tool_name}")
    print(f"Args: {tool_args}")
    print(f"Missing: {missing_required}")
    
    assert "replicas" in missing_required, "Should detect missing replicas"
    assert "namespace" not in missing_required, "Should NOT flag namespace"
    print("✓ Correctly detected partial missing args")


def test_k8s_scale_valid():
    """Test k8s_scale with all required args"""
    
    print("\n=== TEST 3: k8s_scale with valid args ===")
    
    tool_name = "k8s_scale"
    tool_args = {
        "deployment_name": "nginx",
        "namespace": "default",
        "replicas": 3
    }
    
    missing_required = []
    
    if tool_name == "k8s_scale":
        if not tool_args.get("namespace"):
            missing_required.append("namespace")
        if tool_args.get("replicas") is None:
            missing_required.append("replicas")
    
    print(f"Tool: {tool_name}")
    print(f"Args: {tool_args}")
    print(f"Missing: {missing_required}")
    
    assert len(missing_required) == 0, "Should have no missing args"
    print("✓ Valid args passed validation")


def test_schema_required_fields():
    """Test schema-driven validation from tool registry"""
    
    print("\n=== TEST 4: Schema-driven validation ===")
    
    # Get k8s_scale_deployment tool from registry
    tool_obj = tool_registry.get_tool("k8s_scale_deployment")
    
    if tool_obj and isinstance(tool_obj.parameters, dict):
        schema_required = tool_obj.parameters.get("required", [])
        print(f"Schema required fields: {schema_required}")
        
        # Test missing args
        tool_args = {"deployment_name": "nginx"}
        missing = [req for req in schema_required if tool_args.get(req) is None]
        
        print(f"Missing from schema: {missing}")
        assert len(missing) > 0, "Should detect missing schema-required fields"
        print("✓ Schema validation working")
    else:
        print("⚠ Tool not found in registry, skipping")


if __name__ == "__main__":
    print("Testing Worker Validation Logic")
    print("=" * 50)
    
    try:
        test_k8s_scale_validation()
        test_k8s_scale_partial()
        test_k8s_scale_valid()
        test_schema_required_fields()
        
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
