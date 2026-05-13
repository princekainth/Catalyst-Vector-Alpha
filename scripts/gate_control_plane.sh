#!/usr/bin/env bash
set -euo pipefail

python3 -m py_compile cva_runtime/control_plane/capabilities.py
python3 -m py_compile cva_runtime/control_plane/policy.py
python3 -m py_compile cva_runtime/control_plane/audit_log.py
python3 -m py_compile cva_runtime/control_plane/approvals.py
python3 -m py_compile cva_runtime/control_plane/tool_executor.py
python3 -m py_compile cva_runtime/api/routes_ops.py

if python3 -m pytest --version >/dev/null 2>&1; then
  python3 -m pytest tests/control_plane -q
else
  echo "pytest not installed; falling back to unittest discover"
  python3 -m unittest discover -s tests/control_plane -p 'test_*.py' -v
fi
