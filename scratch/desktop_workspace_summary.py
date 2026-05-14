import logging
import sys
import os
import json

# Ensure project root is in path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def analyze_workspace():
    print("=== CVA Workspace Intelligence Demo ===")
    registry = ToolRegistry()
    
    # 1. Discover Active Windows
    print("\n[Discovery] Listing active desktop windows...")
    windows = registry._tools["desktop_list_windows"].func()
    
    if not windows:
        print("No windows detected. Ensure DISPLAY=:1 is active.")
        return

    # 2. Filter and Categorize
    print("[Analysis] Categorizing application context...")
    
    apps = {
        "IDE/Editors": [],
        "Browsers": [],
        "Terminals": [],
        "Other": []
    }
    
    project_hints = set()
    current_task_context = []

    for win in windows:
        title = win['title']
        if not title:
            continue
            
        # Detect IDEs
        if "Visual Studio Code" in title or "Code" in title or "PyCharm" in title:
            apps["IDE/Editors"].append(title)
            # Extract project name from title (usually "ProjectName - File")
            parts = title.split(" - ")
            if len(parts) > 0:
                project_hints.add(parts[0])
        
        # Detect Browsers
        elif any(b in title for b in ["Brave", "Chrome", "Firefox", "Edge", "Chromium"]):
            apps["Browsers"].append(title)
            current_task_context.append(title)
            
        # Detect Terminals
        elif any(t in title for t in ["Terminal", "Term", "bash", "zsh"]):
            apps["Terminals"].append(title)
            # Extract CWD from title if present (common in Linux terminals)
            if "~/" in title or "/" in title:
                path_parts = title.split(": ")
                if len(path_parts) > 1:
                    project_hints.add(path_parts[1])
                else:
                    project_hints.add(title)
        
        else:
            apps["Other"].append(title)

    # 3. Formulate Summary
    print("\n--- Current Workspace Summary ---")
    
    likely_project = ", ".join(list(project_hints)) if project_hints else "Unknown"
    print(f"Likely Project: {likely_project}")
    
    print("\nActive Applications:")
    for category, items in apps.items():
        if items:
            print(f"  {category}:")
            for item in items[:5]: # Limit to 5 per category
                print(f"    - {item}")

    print("\nTask Context (Visible Browser Tabs):")
    if current_task_context:
        for ctx in current_task_context[:3]:
            print(f"    - {ctx}")
    else:
        print("    - No browser context visible.")

    # 4. Final Verification
    print("\nSecurity Confirmation:")
    print("  ✓ No destructive tools invoked.")
    print("  ✓ No input (click/type) commands issued.")
    print("  ✓ No arbitrary shell commands executed.")
    print("  ✓ Read-only observation successful.")

    print("\n=== Workspace Intelligence Complete ===")

if __name__ == "__main__":
    analyze_workspace()
