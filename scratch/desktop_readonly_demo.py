import logging
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_desktop_demo():
    print("=== CVA Desktop Operator v0.1 Read-Only Demo ===")
    registry = ToolRegistry()
    
    # 1. List Windows
    print("\n1. Discovering Active Windows...")
    windows = registry._tools["desktop_list_windows"].func()
    
    if not windows:
        print("No windows detected. Ensure DISPLAY=:1 is active.")
        return

    # Filter for interesting apps
    apps = [w['title'] for w in windows if w['title']]
    print(f"Detected {len(apps)} titled windows.")
    
    workspace_summary = []
    interesting_keywords = ["Brave", "Visual Studio Code", "Code", "Terminal", "Dashboard", "FirstSemanticOS"]
    
    for app in apps:
        if any(kw in app for kw in interesting_keywords):
            workspace_summary.append(app)
            
    print("Working Workspace Summary:")
    for item in workspace_summary[:10]:
        print(f"  [ACTIVE] {item}")

    # 2. Get Details of a window
    if windows:
        target = windows[0]
        print(f"\n2. Inspecting Window: {target['title']} ({target['id']})")
        details = registry._tools["desktop_get_window_details"].func(window_id=target['id'])
        print(f"   Geometry: {details}")

    # 3. Take Screenshot (Validated Path)
    print("\n3. Capturing Workspace Screenshot...")
    # Try capturing the first window if root fails (common in some X11 setups)
    window_id = windows[0]['id'] if windows else None
    res_safe = registry._tools["desktop_take_screenshot"].func(
        path="scratch/demo_window.xwd",
        window_id=window_id
    )
    print(f"   Safe Capture (Window {window_id}): {res_safe}")

    # Malicious path (should fail validation)
    print("\n4. Testing Path Traversal Protection...")
    res_malicious = registry._tools["desktop_take_screenshot"].func(path="scratch/../../etc/shadow")
    # Note: The validator is called by ToolExecutor, but here we call the lambda directly.
    # However, I added the '..' check inside _safe_desktop_screenshot too.
    print(f"   Malicious Capture (Direct Call): {res_malicious}")

    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run_desktop_demo()
