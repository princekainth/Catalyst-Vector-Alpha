import ast
import json
import os
import time
from typing import Any


def find_repo_root(start_path: str) -> str:
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        if os.path.isfile(os.path.join(current, "pyproject.toml")):
            return current
        if os.path.isfile(os.path.join(current, "setup.cfg")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return current
        current = parent


def run_policy_scan(scan_id: str | None, base_dir: str, repo_root: str | None, logger=None) -> dict:
    if not scan_id:
        scan_id = f"unknown-{int(time.time())}"
    if not base_dir:
        base_dir = "persistence_data"
    if not repo_root:
        repo_root = find_repo_root(os.path.dirname(os.path.abspath(__file__)))

    scans_dir = os.path.join(base_dir, "policy_scans")
    os.makedirs(scans_dir, exist_ok=True)

    checks = {
        "planner_tools": "pass",
        "kubectl_polling": "pass",
        "tool_registry": "pass",
        "quarantine": "pass",
        "curiosity_gating": "pass",
    }
    findings: list[dict[str, Any]] = []

    def _fail(check: str, message: str, severity: str = "high") -> None:
        checks[check] = "fail"
        findings.append({"check": check, "severity": severity, "message": message})

    baseline_planner_path = os.path.join(scans_dir, "baseline_planner_tools.json")
    baseline_kubectl_path = os.path.join(scans_dir, "baseline_kubectl_files.json")

    agents_path = os.path.join(repo_root, "agents.py")
    agents_text = _read_text(agents_path)
    planner_tools = _extract_safe_calls(agents_text, "ProtoAgent_Planner")
    if "ProtoAgent_Planner" not in agents_text:
        _fail("planner_tools", "Planner class not found in agents.py.")

    baseline_planner = _load_json(baseline_planner_path)
    if baseline_planner and isinstance(baseline_planner.get("planner_tools"), list):
        base_tools = set(baseline_planner.get("planner_tools") or [])
        current_tools = set(planner_tools)
        added = sorted(current_tools - base_tools)
        removed = sorted(base_tools - current_tools)
        if added or removed:
            _fail(
                "planner_tools",
                f"Planner tool calls changed. Added: {added or 'none'}, Removed: {removed or 'none'}",
            )
    else:
        _write_json(baseline_planner_path, {"planner_tools": planner_tools})

    tool_registry_path = os.path.join(repo_root, "tool_registry.py")
    tool_registry_text = _read_text(tool_registry_path)
    if not tool_registry_text:
        _fail("tool_registry", "tool_registry.py missing or unreadable.")
    else:
        guard_ok = "caller_agent" in tool_registry_text and "planner" in tool_registry_text.lower()
        if not guard_ok:
            _fail("tool_registry", "Planner guard missing in tool_registry.safe_call.")
        if "curiosity" in tool_registry_text.lower():
            _fail("tool_registry", "Curiosity exposure detected in tool_registry.py.")

    quarantine_path = os.path.join(repo_root, "quarantine.py")
    quarantine_text = _read_text(quarantine_path)
    if not quarantine_text:
        _fail("quarantine", "quarantine.py missing or unreadable.")
    else:
        if "def is_quarantined" not in quarantine_text:
            _fail("quarantine", "is_quarantined missing in quarantine.py.")
        if "def set_quarantine" not in quarantine_text:
            _fail("quarantine", "set_quarantine missing in quarantine.py.")
    if not _agents_use_quarantine(agents_text):
        _fail("quarantine", "Observer/Security quarantine checks missing in agents.py.")

    curiosity_path = os.path.join(repo_root, "curiosity_loop.py")
    curiosity_text = _read_text(curiosity_path)
    if not curiosity_text:
        _fail("curiosity_gating", "curiosity_loop.py missing or unreadable.")
    else:
        if not _curiosity_gate_ok(curiosity_text):
            _fail("curiosity_gating", "Curiosity gating checks are incomplete.")

    k8s_stats = _scan_k8s_polling(repo_root)
    baseline_kubectl = _load_json(baseline_kubectl_path)
    if baseline_kubectl and isinstance(baseline_kubectl.get("kubectl_files"), list):
        base_files = set(baseline_kubectl.get("kubectl_files") or [])
        current_files = set(k8s_stats["kubectl_files"])
        new_files = sorted(current_files - base_files)
        if new_files:
            _fail("kubectl_polling", f"New kubectl usage detected: {new_files}")
        base_loop_count = int(baseline_kubectl.get("polling_loop_count", 0) or 0)
        if k8s_stats["polling_loop_count"] > base_loop_count:
            _fail("kubectl_polling", "Polling loop count increased from baseline.")
        base_min = baseline_kubectl.get("min_poll_interval_s")
        if base_min is not None and k8s_stats["min_poll_interval_s"] is not None:
            if float(k8s_stats["min_poll_interval_s"]) < float(base_min):
                _fail("kubectl_polling", "Polling interval decreased from baseline.")
    else:
        baseline_payload = {
            "kubectl_files": k8s_stats["kubectl_files"],
            "polling_loop_count": k8s_stats["polling_loop_count"],
            "min_poll_interval_s": k8s_stats["min_poll_interval_s"],
        }
        _write_json(baseline_kubectl_path, baseline_payload)

    status = "pass" if all(v == "pass" for v in checks.values()) else "fail"
    report = {
        "scan_id": scan_id,
        "status": status,
        "checks": checks,
        "findings": findings,
        "timestamp": int(time.time()),
        "baseline": {"planner_tools": planner_tools},
    }

    out_path = os.path.join(scans_dir, f"{scan_id}.json")
    _write_json(out_path, report)

    if logger:
        if status == "pass":
            logger.info(f"[PolicyScan] PASS scan_id={scan_id}")
        else:
            logger.info(f"[PolicyScan] FAIL scan_id={scan_id} findings={len(findings)}")

    return report


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _load_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json(path: str, payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
    except Exception:
        pass


def _extract_safe_calls(text: str, class_name: str) -> list[str]:
    if not text:
        return []
    try:
        tree = ast.parse(text)
    except Exception:
        return []
    class_node = _find_class(tree, class_name)
    if not class_node:
        return []
    calls = set()
    for node in ast.walk(class_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "safe_call":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                calls.add(node.args[0].value)
    return sorted(calls)


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _agents_use_quarantine(text: str) -> bool:
    if not text:
        return False
    try:
        tree = ast.parse(text)
    except Exception:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in {"is_quarantined", "_is_quarantined"}:
                return True
    return False


def _curiosity_gate_ok(text: str) -> bool:
    if not text:
        return False
    try:
        tree = ast.parse(text)
    except Exception:
        return False
    cls = _find_class(tree, "CuriosityLoop")
    if not cls:
        return False
    should_run = None
    loop_fn = None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_should_run":
            should_run = node
        if isinstance(node, ast.FunctionDef) and node.name == "_loop":
            loop_fn = node
    if not should_run or not loop_fn:
        return False
    required_calls = {"_event_gate_idle", "_get_cpu_usage", "_k8s_all_clear", "_incidents_quiet"}
    seen = set()
    for node in ast.walk(should_run):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in required_calls:
                seen.add(name)
    if seen != required_calls:
        return False
    loop_calls = set()
    for node in ast.walk(loop_fn):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name:
                loop_calls.add(name)
    if "_should_run" not in loop_calls:
        return False
    return True


def _scan_k8s_polling(repo_root: str) -> dict:
    kubectl_files: set[str] = set()
    loop_count = 0
    min_sleep = None
    skip_dirs = {".git", "__pycache__", "venv", ".venv", "node_modules", "logs", "persistence_data", "tests"}
    markers = ("k8s", "kubectl", "get_pod_status", "watch_k8s", "kubernetes")

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            rel_path = os.path.relpath(path, repo_root)
            text = _read_text(path)
            if not text:
                continue
            try:
                tree = ast.parse(text)
            except Exception:
                tree = None

            if _contains_kubectl(text, tree):
                kubectl_files.add(rel_path)

            k8s_related = "k8s" in rel_path.lower() or any(m in text for m in markers)
            if not (k8s_related and tree):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.While) and _is_true(node.test):
                    loop_sleep = _min_sleep_in_loop(node)
                    if loop_sleep is not None:
                        loop_count += 1
                        if min_sleep is None or loop_sleep < min_sleep:
                            min_sleep = loop_sleep

    return {
        "kubectl_files": sorted(kubectl_files),
        "polling_loop_count": loop_count,
        "min_poll_interval_s": min_sleep,
    }


def _contains_kubectl(text: str, tree: ast.AST | None) -> bool:
    if "kubectl" not in text:
        return False
    if not tree:
        return "kubectl" in text
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "kubectl":
            return True
        if isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and elt.value == "kubectl":
                    return True
    return False


def _is_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _min_sleep_in_loop(loop: ast.While) -> float | None:
    min_sleep = None
    for node in ast.walk(loop):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in {"sleep"}:
                if node.args:
                    val = _const_number(node.args[0])
                    if val is not None and (min_sleep is None or val < min_sleep):
                        min_sleep = val
    return min_sleep


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _const_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None
