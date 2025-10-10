# swarm_console.py
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import sys
import time
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------
# Paths & constants (align with your project)
# --------------------------------------------------------------------
PERSISTENCE_DIR = os.getenv("PERSISTENCE_DIR", "persistence_data")
SWARM_STATE_FILE = os.path.join(PERSISTENCE_DIR, "swarm_state.json")
PAUSED_AGENTS_FILE = os.path.join(PERSISTENCE_DIR, "paused_agents.json")
SWARM_ACTIVITY_LOG = os.path.join(PERSISTENCE_DIR, "swarm_activity.jsonl")

os.makedirs(PERSISTENCE_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Small utilities: robust JSON IO, human prints
# --------------------------------------------------------------------
def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        # tolerate corruptions; never block console
        return default
    except Exception:
        return default

def _write_json_atomic(path: str, data: Any) -> None:
    # write to tmp in same dir, then atomic replace
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        with io.open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        # best-effort: try regular write as fallback
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def _agent_state_path(agent_name: str) -> str:
    return os.path.join(PERSISTENCE_DIR, f"agent_state_{agent_name}.json")

def _list_agent_names() -> List[str]:
    names: List[str] = []
    for fn in os.listdir(PERSISTENCE_DIR):
        if fn.startswith("agent_state_") and fn.endswith(".json"):
            names.append(fn[len("agent_state_"):-len(".json")])
    names.sort()
    return names

# --------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------
def load_paused_agents() -> List[str]:
    data = _read_json(PAUSED_AGENTS_FILE, default=[])
    return data if isinstance(data, list) else []

def save_paused_agents(paused: Iterable[str]) -> None:
    unique_sorted = sorted(set(paused))
    _write_json_atomic(PAUSED_AGENTS_FILE, unique_sorted)

def load_agent_state(agent_name: str) -> Optional[Dict[str, Any]]:
    path = _agent_state_path(agent_name)
    if not os.path.exists(path):
        return None
    data = _read_json(path, default=None)
    return data if isinstance(data, dict) else None

def load_swarm_state() -> Optional[Dict[str, Any]]:
    if not os.path.exists(SWARM_STATE_FILE):
        return None
    data = _read_json(SWARM_STATE_FILE, default=None)
    return data if isinstance(data, dict) else None

# --------------------------------------------------------------------
# Console core actions
# --------------------------------------------------------------------
def action_list_agents(json_mode: bool) -> int:
    names = _list_agent_names()
    paused = set(load_paused_agents())

    if json_mode:
        payload = [{"name": n, "status": "PAUSED" if n in paused else "ACTIVE"} for n in names]
        print(json.dumps(payload, indent=2))
        return 0

    print("\n--- Active Agents ---")
    if not names:
        print("No active agents found in persistence_data. Start your orchestrator first.")
        return 0
    for n in names:
        tag = "[PAUSED]" if n in paused else "[ACTIVE]"
        print(f"- {n} {tag}")
    return 0

def _print_agent_status_human(agent_name: str, state: Dict[str, Any]) -> None:
    print(f"\n--- Status for Agent: {agent_name} ---")
    role = state.get("eidos_spec", {}).get("role", "N/A")
    print(f"  Role: {role}")
    print(f"  Current Intent: {state.get('current_intent', 'N/A')}")
    print(f"  Location: {state.get('location', 'N/A')}")
    membership = state.get("swarm_membership") or []
    print(f"  Swarm Membership: {', '.join(membership) if membership else 'None'}")

    grad = state.get("sovereign_gradient")
    if grad:
        print("  Sovereign Gradient:")
        print(f"    Vector: {grad.get('autonomy_vector', 'N/A')}")
        ec = grad.get("ethical_constraints") or []
        print(f"    Ethical Constraints: {', '.join(ec) if ec else 'None'}")
    else:
        print("  Sovereign Gradient: Not set")

    paused = set(load_paused_agents())
    print(f"  Console Status: {'PAUSED' if agent_name in paused else 'RUNNING'}")
    print("-----------------------------------")

def action_get_status(target: str, json_mode: bool) -> int:
    # agent?
    if os.path.exists(_agent_state_path(target)):
        st = load_agent_state(target)
        if not st:
            print(f"Error: Could not load state for agent '{target}'.")
            return 1
        if json_mode:
            out = {"entity": "agent", "name": target, "state": st}
            print(json.dumps(out, indent=2))
        else:
            _print_agent_status_human(target, st)
        return 0

    # swarm?
    if target == "AlphaEcoSwarm":
        st = load_swarm_state()
        if not st:
            print(f"Error: Swarm state not found at '{SWARM_STATE_FILE}'.")
            return 1
        if json_mode:
            out = {"entity": "swarm", "name": st.get("name", "AlphaEcoSwarm"), "state": st}
            print(json.dumps(out, indent=2))
        else:
            print(f"\n--- Status for Swarm: {st.get('name', 'AlphaEcoSwarm')} ---")
            print(f"  Goal: {st.get('goal', 'N/A')}")
            members = st.get("members") or []
            print(f"  Members: {', '.join(members) if members else 'None'}")
            print(f"  Consensus Mechanism: {st.get('consensus_mechanism', 'N/A')}")
            print(f"  Description: {st.get('description', 'N/A')}")
            grad = st.get("sovereign_gradient")
            if grad:
                print("  Sovereign Gradient:")
                print(f"    Vector: {grad.get('autonomy_vector', 'N/A')}")
                ec = grad.get("ethical_constraints") or []
                print(f"    Ethical Constraints: {', '.join(ec) if ec else 'None'}")
                print(f"    Override Threshold: {grad.get('override_threshold', 'N/A')}")
            else:
                print("  Sovereign Gradient: Not set")
            print("-----------------------------------")
        return 0

    print(f"Error: '{target}' is not an agent or the supported swarm 'AlphaEcoSwarm'.")
    return 1

def action_get_memory(target: str, json_mode: bool) -> int:
    if os.path.exists(_agent_state_path(target)):
        st = load_agent_state(target) or {}
        mems = st.get("memetic_kernel", {}).get("memories") or []
        if json_mode:
            print(json.dumps({"entity": "agent", "name": target, "memories": mems[-20:]}, indent=2))
            return 0

        print(f"\n--- Recent Memories for Agent: {target} ---")
        if not mems:
            print("  No memories recorded yet.")
            print("----------------------------------")
            return 0
        for m in mems[-5:]:
            ts = m.get("timestamp", "N/A")
            mt = m.get("type", "N/A")
            content = m.get("content", "N/A")
            print(f"  [{ts}] <{mt}> {content}")
        print("----------------------------------")
        return 0

    if target == "AlphaEcoSwarm":
        st = load_swarm_state() or {}
        mems = st.get("memetic_kernel", {}).get("memories") or []
        if json_mode:
            print(json.dumps({"entity": "swarm", "name": "AlphaEcoSwarm", "memories": mems[-20:]}, indent=2))
            return 0

        print(f"\n--- Recent Memories for Swarm: {target} ---")
        if not mems:
            print("  No memories recorded yet.")
            print("----------------------------------")
            return 0
        for m in mems[-5:]:
            ts = m.get("timestamp", "N/A")
            mt = m.get("type", "N/A")
            content = m.get("content", "N/A")
            print(f"  [{ts}] <{mt}> {content}")
        print("----------------------------------")
        return 0

    print(f"Error: '{target}' is not recognized.")
    return 1

def action_broadcast_intent(target: str, new_intent: str, json_mode: bool) -> int:
    # Only allow known agent or the main swarm
    is_agent = os.path.exists(_agent_state_path(target))
    is_swarm = (target == "AlphaEcoSwarm")

    if not (is_agent or is_swarm):
        print(f"Error: Broadcast target '{target}' not found or not supported.")
        return 1

    payload = {
        "target": target,
        "new_intent": new_intent,
        "timestamp": _now_iso(),
        "status": "pending",
    }
    dst = os.path.join(PERSISTENCE_DIR, f"intent_override_{target}.json")
    _write_json_atomic(dst, payload)

    if json_mode:
        print(json.dumps({"queued": True, "path": dst, "payload": payload}, indent=2))
    else:
        print(f"Broadcast queued: '{new_intent}' → {target}. (written {dst})")
    return 0

def action_pause(agent_names: List[str], json_mode: bool) -> int:
    existing = set(load_paused_agents())
    all_agents = set(_list_agent_names())
    updated = []
    for name in agent_names:
        if name not in all_agents:
            print(f"Warning: Agent '{name}' not found; skipping.")
            continue
        if name not in existing:
            existing.add(name)
            updated.append(name)
    save_paused_agents(existing)

    if json_mode:
        print(json.dumps({"paused": sorted(updated)}, indent=2))
    else:
        for n in updated:
            print(f"Agent '{n}' paused.")
    return 0

def action_resume(agent_names: List[str], json_mode: bool) -> int:
    existing = set(load_paused_agents())
    updated = []
    for name in agent_names:
        if name in existing:
            existing.remove(name)
            updated.append(name)
    save_paused_agents(existing)

    if json_mode:
        print(json.dumps({"resumed": sorted(updated)}, indent=2))
    else:
        for n in updated:
            print(f"Agent '{n}' resumed.")
    return 0

def action_watch_log(json_mode: bool, follow: bool, max_lines: int) -> int:
    """
    Print last N entries of the JSONL log; optionally follow (tail -f).
    """
    path = SWARM_ACTIVITY_LOG
    if not os.path.exists(path):
        print(f"{path} not found. Start your monitor/orchestrator first.")
        return 1

    def _iter_entries() -> Iterable[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            # seek to last N lines cheaply
            if max_lines > 0:
                try:
                    f.seek(0, os.SEEK_END)
                    pos = f.tell()
                    block = 4096
                    data = b""
                    while pos > 0 and data.count(b"\n") <= max_lines:
                        read = min(block, pos)
                        pos -= read
                        f.seek(pos, os.SEEK_SET)
                        data = f.read(read).encode("utf-8") + data
                    lines = data.decode("utf-8", errors="ignore").splitlines()[-max_lines:]
                except Exception:
                    f.seek(0)
                    lines = f.readlines()[-max_lines:]
            else:
                lines = f.readlines()

            for ln in lines:
                try:
                    yield json.loads(ln.strip())
                except Exception:
                    continue

            if not follow:
                return

            # tail
            while True:
                ln = f.readline()
                if not ln:
                    time.sleep(0.3)
                    continue
                try:
                    yield json.loads(ln.strip())
                except Exception:
                    continue

    for entry in _iter_entries():
        if json_mode:
            print(json.dumps(entry, indent=2))
        else:
            ts = entry.get("timestamp", "N/A")
            ev = entry.get("event_type", "N/A")
            src = entry.get("source", "N/A")
            desc = entry.get("description", "N/A")
            details = entry.get("details", {})
            dstr = json.dumps(details, indent=2) if details else "{}"
            print(f"[{ts}] <{ev}> {src}\n  {desc}\n  Details: {dstr}\n---")
    return 0

def action_latest_snapshot(json_mode: bool) -> int:
    """
    Read the last SWARM_SNAPSHOT from the log (if any) and display summary.
    """
    path = SWARM_ACTIVITY_LOG
    if not os.path.exists(path):
        print(f"{path} not found.")
        return 1

    last_snap = None
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                obj = json.loads(ln.strip())
                if obj.get("event_type") == "SWARM_SNAPSHOT":
                    last_snap = obj
            except Exception:
                continue

    if not last_snap:
        print("No snapshot entries yet.")
        return 0

    if json_mode:
        print(json.dumps(last_snap, indent=2))
        return 0

    det = last_snap.get("details", {})
    agents = det.get("agents", {})
    p99 = det.get("p99_latency_per_agent", {})
    sys_ = det.get("system", {})

    print("--- Latest Snapshot ---")
    print(f"Timestamp: {last_snap.get('timestamp')}")
    print(f"Agents ({len(agents)}):")
    for a, st in agents.items():
        print(f"  - {a}: {st}")
    if p99:
        print("p99 latency (s):")
        for a, v in p99.items():
            print(f"  - {a}: {v:.2f}")
    print("System EWMA:")
    print(f"  proc_cpu={sys_.get('proc_cpu_ewma', 0):.1f}%  proc_mem={sys_.get('proc_mem_ewma', 0):.1f}%")
    print(f"  sys_cpu={sys_.get('sys_cpu_ewma', 0):.1f}%   sys_mem={sys_.get('sys_mem_ewma', 0):.1f}%")
    print("-----------------------")
    return 0

# --------------------------------------------------------------------
# CLI arg parsing
# --------------------------------------------------------------------
@dataclass
class Args:
    json: bool = False

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swarm-console",
        description="Swarm Console: inspect and control your agent swarm.",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human text.")

    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list-agents", help="List all active agents.")

    gs = sub.add_parser("get-status", help="Show detailed status of an agent or the swarm.")
    gs.add_argument("target", help="Agent name or 'AlphaEcoSwarm'")

    gm = sub.add_parser("get-memory", help="Show recent memories for an agent or the swarm.")
    gm.add_argument("target", help="Agent name or 'AlphaEcoSwarm'")

    bc = sub.add_parser("broadcast-intent", help="Broadcast a new intent to an agent/swarm.")
    bc.add_argument("target", help="Agent name or 'AlphaEcoSwarm'")
    bc.add_argument("intent", help="New intent string")

    pause = sub.add_parser("pause", help="Pause one or more agents.")
    pause.add_argument("agent", nargs="+", help="Agent name(s)")

    resume = sub.add_parser("resume", help="Resume one or more agents.")
    resume.add_argument("agent", nargs="+", help="Agent name(s)")

    watch = sub.add_parser("watch", help="Tail/swipe the swarm activity log.")
    watch.add_argument("-n", "--lines", type=int, default=50, help="Print last N lines before following.")
    watch.add_argument("-f", "--follow", action="store_true", help="Follow the log (tail -f).")

    sub.add_parser("latest", help="Show latest SWARM_SNAPSHOT summary.")

    repl = sub.add_parser("repl", help="Interactive console (classic commands).")

    return p

# --------------------------------------------------------------------
# Interactive REPL (keeps your original verbs)
# --------------------------------------------------------------------
def run_repl(json_mode: bool) -> int:
    print("--- Welcome to the Swarm Console (REPL) ---")
    print("Type 'help' for commands, 'exit' to quit.")
    while True:
        try:
            line = input("\nswarm-console> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            return 0
        if line.lower() == "help":
            print("\n--- Commands ---")
            print("  list agents")
            print("  get status <agent|AlphaEcoSwarm>")
            print("  get memory <agent|AlphaEcoSwarm>")
            print("  broadcast intent <target>:<new_intent>")
            print("  pause agent <agent> [<agent> ...]")
            print("  resume agent <agent> [<agent> ...]")
            print("  watch [-n N] [-f]")
            print("  latest")
            print("  exit")
            continue

        # minimal parser for backward-compatible verbs
        parts = line.split()
        if parts[:2] == ["list", "agents"]:
            action_list_agents(json_mode)
            continue

        if parts[:2] == ["get", "status"] and len(parts) >= 3:
            action_get_status(" ".join(parts[2:]), json_mode)
            continue

        if parts[:2] == ["get", "memory"] and len(parts) >= 3:
            action_get_memory(" ".join(parts[2:]), json_mode)
            continue

        if parts[:2] == ["broadcast", "intent"] and len(parts) >= 3:
            if ":" in parts[2]:
                target, intent = parts[2].split(":", 1)
                action_broadcast_intent(target.strip(), intent.strip(), json_mode)
            else:
                print("Usage: broadcast intent <target>:<new_intent>")
            continue

        if parts[0:2] == ["pause", "agent"] and len(parts) >= 3:
            action_pause(parts[2:], json_mode)
            continue

        if parts[0:2] == ["resume", "agent"] and len(parts) >= 3:
            action_resume(parts[2:], json_mode)
            continue

        if parts[0] == "watch":
            # basic flags support in REPL: watch -n 100 -f
            n = 50
            follow = False
            if "-n" in parts:
                try:
                    idx = parts.index("-n")
                    n = int(parts[idx + 1])
                except Exception:
                    pass
            if "-f" in parts:
                follow = True
            action_watch_log(json_mode, follow=follow, max_lines=n)
            continue

        if parts[0] == "latest":
            action_latest_snapshot(json_mode)
            continue

        print(f"Unknown command: '{line}'. Type 'help' for commands.")
    # end while

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # default: repl if no subcommand
    if not args.cmd:
        return run_repl(json_mode=args.json)

    if args.cmd == "repl":
        return run_repl(json_mode=args.json)
    if args.cmd == "list-agents":
        return action_list_agents(json_mode=args.json)
    if args.cmd == "get-status":
        return action_get_status(args.target, json_mode=args.json)
    if args.cmd == "get-memory":
        return action_get_memory(args.target, json_mode=args.json)
    if args.cmd == "broadcast-intent":
        return action_broadcast_intent(args.target, args.intent, json_mode=args.json)
    if args.cmd == "pause":
        return action_pause(args.agent, json_mode=args.json)
    if args.cmd == "resume":
        return action_resume(args.agent, json_mode=args.json)
    if args.cmd == "watch":
        return action_watch_log(json_mode=args.json, follow=args.follow, max_lines=args.lines)
    if args.cmd == "latest":
        return action_latest_snapshot(json_mode=args.json)

    parser.print_help()
    return 2

if __name__ == "__main__":
    sys.exit(main())
