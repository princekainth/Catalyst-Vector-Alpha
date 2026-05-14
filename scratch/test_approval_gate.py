import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry

def test_approval_gate():
    print("=== CVA Destructive Approval Gate Test ===")
    registry = ToolRegistry()
    
    # 1. Create a file
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/a.txt", "w") as f:
        f.write("hello")
    print("Created scratch/a.txt")

    # 2. Try to move it (Should be GATED)
    print("Attempting to move scratch/a.txt to scratch/b.txt...")
    res = registry._tools["desktop_move_file"].func(
        source="scratch/a.txt",
        destination="scratch/b.txt"
    )
    print(f"Result: {res}")
    
    # 3. Verify file did NOT move
    if os.path.exists("scratch/a.txt") and not os.path.exists("scratch/b.txt"):
        print("✅ SUCCESS: File did not move. Gate is active.")
    else:
        print("❌ FAILURE: File moved or source missing!")

if __name__ == "__main__":
    test_approval_gate()
