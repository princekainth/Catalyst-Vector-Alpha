import inspect
import tools_system, tools_k8s, tools_security, tools_web, tools_swarm

all_tools = ['ToolConfig', 'standardize_response', 'ToolCache']

custom_exports = [
    'get_pod_status', 'check_network_connectivity', 'watch_k8s_events', 
    'watch_k8s_audit_events', 'microsoft_autonomous_remediation', 
    'collect_imagepull_forensics', 'analyze_imagepull_failure', 
    'execute_imagepull_remediation', 'reply_to_user', 'remember_event', 
    'search_memory', 'capture_system_screenshot', 'tune_hyperparameters', 
    'self_patch', 'spawn_agent'
]

modules = [tools_system, tools_k8s, tools_security, tools_web, tools_swarm]
for module in modules:
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and getattr(obj, '__module__', '') == module.__name__:
            if name.endswith('_tool') or name in custom_exports:
                all_tools.append(name)

all_tools = sorted(list(set(all_tools)))
print('Found', len(all_tools), 'tools')

with open('tools_impl.py', 'r') as f:
    content = f.read()

import re
new_content = re.sub(r'__all__\s*=\s*\[.*?\]', '__all__ = ' + repr(all_tools), content, flags=re.DOTALL)

with open('tools_impl.py', 'w') as f:
    f.write(new_content)

print("Updated tools_impl.py successfully.")
