
import os
import sys
sys.path.append(os.getcwd())

def verify_api_approve_safety():
    from app import app
    client = app.test_client()
    
    print("--- Testing /api/approve with non-existent trace ---")
    res = client.post("/api/approve", json={
        "task_id": "trc_fake_123",
        "approval_token": "tok_some_token"
    })
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.get_json()}")
    
    if res.status_code == 404:
        print("✓ SUCCESS: Non-existent trace rejected.")
    else:
        print("FAIL: Expected 404 for fake trace.")

if __name__ == "__main__":
    verify_api_approve_safety()
