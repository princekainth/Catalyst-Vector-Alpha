import os
import re

def split_tools():
    with open("tools_impl.py", "r") as f:
        content = f.read()

    # Define the markers
    sys_marker = "# ---- System / Local Tools ----"
    k8s_marker = "# ---- Kubernetes Tools ----"
    sec_marker = "# ---- Security / Networking Tools ----"
    web_marker = "# ---- Environment / World / Knowledge Tools ----"
    mgmt_marker = "# ---- Management & Utility Tools ----"

    core_part = content.split(sys_marker)[0]
    
    rest = content.split(sys_marker)[1]
    sys_part = sys_marker + "\n" + rest.split(k8s_marker)[0]
    
    rest = rest.split(k8s_marker)[1]
    k8s_part = k8s_marker + "\n" + rest.split(sec_marker)[0]
    
    rest = rest.split(sec_marker)[1]
    sec_part = sec_marker + "\n" + rest.split(web_marker)[0]
    
    rest = rest.split(web_marker)[1]
    web_part = web_marker + "\n" + rest.split(mgmt_marker)[0]
    
    rest = rest.split(mgmt_marker)[1]
    swarm_part = mgmt_marker + "\n" + rest # includes all subsequent sections (memory, agents, phase 8)

    # We need to add all the core imports to the top of each new file.
    # We can extract the imports from core_part.
    # For safety, let's just prepend the entire core_part (minus the ToolConfig validation at the end, though that is safe) 
    # Actually, importing tools_impl from these files is better, but tools_impl will import them, causing a circular import.
    # Better: Put the imports in each file. A simple way is to just put the raw core imports.
    
    # Let's extract imports and globals from core_part.
    lines = core_part.split('\n')
    import_lines = []
    for line in lines:
        if line.startswith("from ") or line.startswith("import ") or line.startswith("try:") or line.startswith("except ") or line.startswith("    "):
            import_lines.append(line)
        elif line.startswith("_") or line.startswith("logger") or line.startswith("class ") or line.startswith("def ") or line.startswith("    ") or line.startswith("    #"):
            import_lines.append(line)
        elif line.strip() == "":
            import_lines.append(line)
        elif line.startswith("#"):
            import_lines.append(line)
            
    core_content_for_submodules = core_part

    # Write tools_system.py
    with open("tools_system.py", "w") as f:
        f.write(core_content_for_submodules + "\n" + sys_part)

    # Write tools_k8s.py
    with open("tools_k8s.py", "w") as f:
        f.write(core_content_for_submodules + "\n" + k8s_part)

    # Write tools_security.py
    with open("tools_security.py", "w") as f:
        f.write(core_content_for_submodules + "\n" + sec_part)

    # Write tools_web.py
    with open("tools_web.py", "w") as f:
        f.write(core_content_for_submodules + "\n" + web_part)

    # Write tools_swarm.py
    with open("tools_swarm.py", "w") as f:
        f.write(core_content_for_submodules + "\n" + swarm_part)

    # Finally, rewrite tools_impl.py to just re-export everything.
    new_tools_impl = core_part + """
# ------------------------------------------------------------------------------
# Submodule Imports
# ------------------------------------------------------------------------------
from tools_system import *
from tools_k8s import *
from tools_security import *
from tools_web import *
from tools_swarm import *

# Initialize configuration validation on import
ToolConfig.validate()
"""
    
    # Keep the final __all__ list from the original tools_impl.py
    all_list_marker = "__all__ = ["
    if all_list_marker in content:
        all_section = all_list_marker + content.split(all_list_marker)[1]
        new_tools_impl += "\n" + all_section

    with open("tools_impl.py", "w") as f:
        f.write(new_tools_impl)

if __name__ == "__main__":
    split_tools()
