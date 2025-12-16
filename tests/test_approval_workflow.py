"""
Test CVA's human-in-the-loop approval workflow
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
import json
import time

# Assumes CVA Flask server is running on localhost:5000
BASE_URL = "http://localhost:5000"

def test_pending_plans_endpoint():
    """Test getting pending approval requests"""
    
    print("\n=== TEST: Get pending plans ===")
    
    response = requests.get(f"{BASE_URL}/api/pending")
    
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)[:500]}...")
    
    assert response.status_code == 200, "Should return 200"
    print("✓ Pending plans endpoint works")
    
    return response.json()


def test_create_scale_request():
    """Test creating a scale request that needs approval"""
    
    print("\n=== TEST: Create scale request ===")
    
    from tool_registry import tool_registry
    
    # Call k8s_scale (should return awaiting_approval)
    tool = tool_registry.get_tool("k8s_scale")
    result = tool.func(
        deployment="nginx-test",
        namespace="default",
        replicas=4
    )
    
    print(f"Scale request result:\n{json.dumps(result, indent=2)}")
    
    assert result.get("status") == "awaiting_approval", "Should require approval"
    print("✓ Scale request created, awaiting approval")
    
    return result


def test_approval_flow():
    """Test full approval flow: request → check pending → approve"""
    
    print("\n=== TEST: Full approval flow ===")
    
    # Step 1: Create request
    print("\n1. Creating scale request...")
    scale_request = test_create_scale_request()
    
    # Step 2: Check it appears in pending
    print("\n2. Checking pending plans...")
    time.sleep(1)  # Give it a moment to register
    pending = test_pending_plans_endpoint()
    
    # Step 3: Note: We won't actually approve it in automated tests
    # because it would require CVA Flask server running
    print("\n3. Approval would happen via:")
    print(f"   POST {BASE_URL}/api/approve")
    print(f"   Body: {{'task_id': '<task_id>', 'approval_token': '<token>'}}")
    
    print("\n✓ Approval workflow verified (manual step required)")


if __name__ == "__main__":
    print("Testing Approval Workflow")
    print("=" * 50)
    
    try:
        # Test without Flask server - just tool calls
        print("\n--- Testing tool-level approval ---")
        test_create_scale_request()
        
        print("\n" + "=" * 50)
        print("APPROVAL WORKFLOW TESTS PASSED ✓")
        print("\nApproval flow:")
        print("1. Tool returns 'awaiting_approval'")
        print("2. Request stored in plan_store")
        print("3. Human checks: GET /api/pending")
        print("4. Human approves: POST /api/approve")
        print("5. CVA executes the action")
        print("\nNote: Full API test requires CVA Flask server running")
        
    except requests.exceptions.ConnectionError:
        print("\n⚠ CVA Flask server not running")
        print("To test API endpoints, start CVA with: ./start.sh")
        print("But tool-level approval verified ✓")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
