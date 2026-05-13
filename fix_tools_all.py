import ast

with open("tools_impl.py", "r") as f:
    source = f.read()

tree = ast.parse(source)

defined_names = set()
for node in tree.body:
    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef) or isinstance(node, ast.ClassDef):
        defined_names.add(node.name)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                defined_names.add(target.id)

import tools_impl
real_names = set(dir(tools_impl))

valid_all = []
missing = []
for name in tools_impl.__all__:
    if name in real_names or name in defined_names:
        valid_all.append(name)
    else:
        missing.append(name)

print("Missing from __all__:", missing)

# Now rewrite the file to update __all__
import re
new_source = re.sub(
    r"__all__\s*=\s*\[(.*?)\]", 
    "__all__ = " + repr(valid_all), 
    source, 
    flags=re.DOTALL
)

with open("tools_impl.py", "w") as f:
    f.write(new_source)

print("Fixed tools_impl.py __all__ list")
