import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry

def verify_safety():
    print("=== CVA Desktop Hands Safety Verification ===")
    registry = ToolRegistry()
    hands = registry._get_desktop_hands()
    
    test_cases = [
        # Expected Blocked
        ("~/.ssh/authorized_keys", False),
        ("~/.env", False),
        ("~/.config/test", False),
        ("/etc/passwd", False),
        ("/var/log/test", False),
        ("../escape.txt", False),
        
        # Expected Allowed
        ("scratch/cva_test_note.md", True),
        ("~/Desktop/cva_note.txt", True),
        ("~/Documents/cva_report.md", True)
    ]
    
    passed = 0
    for path, expected_allowed in test_cases:
        result = hands._validate_path(path)
        status = "PASS" if result == expected_allowed else "FAIL"
        print(f"[{status}] Path: {path} | Expected Allowed: {expected_allowed} | Result: {result}")
        if status == "PASS":
            passed += 1
            
    print(f"\nSummary: {passed}/{len(test_cases)} path tests passed.")

    # Symlink Test
    print("\n--- Symlink Escape Test ---")
    os.makedirs("scratch/test_dir", exist_ok=True)
    symlink_path = "scratch/test_dir/passwd_link"
    if os.path.exists(symlink_path):
        os.remove(symlink_path)
    
    try:
        os.symlink("/etc/passwd", symlink_path)
        result = hands._validate_path(symlink_path)
        print(f"Result for symlink to /etc/passwd: {result} (Expected: False)")
        if not result:
            print("[PASS] Symlink escape blocked.")
        else:
            print("[FAIL] Symlink escape ALLOWED!")
    except Exception as e:
        print(f"Symlink creation failed: {e} (Likely permission issue, which is also safe)")

    # Cleanup
    if os.path.exists(symlink_path):
        os.remove(symlink_path)

if __name__ == "__main__":
    verify_safety()
