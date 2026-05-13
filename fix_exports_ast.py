import ast

files = ["tools_system.py", "tools_k8s.py", "tools_security.py", "tools_web.py", "tools_swarm.py"]
all_tools = ['ToolConfig', 'standardize_response', 'ToolCache']
custom_exports = [
    'get_pod_status', 'check_network_connectivity', 'watch_k8s_events', 
    'watch_k8s_audit_events', 'microsoft_autonomous_remediation', 
    'collect_imagepull_forensics', 'analyze_imagepull_failure', 
    'execute_imagepull_remediation', 'reply_to_user', 'remember_event', 
    'search_memory', 'capture_system_screenshot', 'tune_hyperparameters', 
    'self_patch', 'spawn_agent', 'export_system_state_tool'
]

for filename in files:
    with open(filename, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if name.endswith("_tool") or name in custom_exports:
                all_tools.append(name)

all_tools = sorted(list(set(all_tools)))

with open("tools_impl.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("__all__ = ["):
        continue  # skip old __all__ if it exists on one line
    new_lines.append(line)

new_lines.append("\n__all__ = " + repr(all_tools) + "\n")

with open("tools_impl.py", "w") as f:
    f.writelines(new_lines)

print(f"Added __all__ with {len(all_tools)} items successfully.")
