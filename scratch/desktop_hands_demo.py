import logging
import sys
import os
import shutil

# Ensure project root is in path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def test_desktop_hands():
    print("=== CVA Desktop Hands v0.1 Demo ===")
    registry = ToolRegistry()
    
    # Ensure scratch exists
    os.makedirs("scratch", exist_ok=True)
    
    # 1. Create a note (CAUTION)
    # Note: In CVA_DEMO_MODE, most things wait for approval unless APPROVAL_MODE=auto
    print("\n1. Creating a Workspace Note...")
    res_note = registry._tools["desktop_create_note"].func(
        path="scratch/workspace_summary.md",
        content="# Workspace Intelligence\nDetected Brave, VS Code, and Terminal."
    )
    print(f"   Result: {res_note}")

    # 2. Safety Test: Attempt to write to .ssh (BLOCKED)
    print("\n2. Safety Test: Writing to .ssh/authorized_keys (Should FAIL)...")
    res_evil = registry._tools["desktop_write_text_file"].func(
        path="~/.ssh/authorized_keys",
        content="malicious key"
    )
    print(f"   Result: {res_evil}")

    # 3. Propose a move (DESTRUCTIVE - Awaiting Approval)
    print("\n3. Proposing a File Move (Requires Approval)...")
    res_move = registry._tools["desktop_move_file"].func(
        source="scratch/workspace_summary.md",
        destination="scratch/workspace_summary_archived.md"
    )
    print(f"   Result: {res_move}")

    # 4. Patch a file (DESTRUCTIVE - Awaiting Approval)
    print("\n4. Proposing a Text Patch (Requires Approval)...")
    res_patch = registry._tools["desktop_modify_text_file"].func(
        path="scratch/workspace_summary.md",
        search="Brave",
        replace="Brave Browser"
    )
    print(f"   Result: {res_patch}")

    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    test_desktop_hands()
