# swarm_monitor.py
from __future__ import annotations

import json
import os
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Deque, Callable, Optional, List, Any, Tuple, Set
from collections import defaultdict, deque
from statistics import mean

import psutil

try:
    # Optional: if present, we’ll expose Prometheus metrics (no hard dependency).
    from prometheus_client import Gauge, start_http_server  # type: ignore
    _PROM_AVAILABLE = True
except Exception:
    _PROM_AVAILABLE = False

# Your registry + tools (used for interventions)
from tool_registry import tool_registry

# ------------------------------------------------------------------------------
# Defaults / paths
# ------------------------------------------------------------------------------
PERSISTENCE_DIR = os.getenv("PERSISTENCE_DIR", "persistence_data")
SWARM_ACTIVITY_LOG = os.path.join(PERSISTENCE_DIR, "swarm_activity.jsonl")
os.makedirs(PERSISTENCE_DIR, exist_ok=True)

logger = logging.getLogger("CatalystLogger")

# ------------------------------------------------------------------------------
# Config dataclasses
# ------------------------------------------------------------------------------
@dataclass
class ResourceThresholds:
    cpu_high_pct: float = 90.0
    mem_high_pct: float = 85.0
    task_p99_latency_s: float = 5.0  # "slow task" guidance

@dataclass
class StagnationPolicy:
    idle_intents: Set[str] = field(default_factory=lambda: {
        "Awaiting tasks from the Planner.",
        "Executing injected plan directives.",
        "Idle",  # allow explicit Idle
    })
    idle_ratio_threshold: float = 0.9     # % of agents idle to count an idle cycle
    consecutive_idle_cycles: int = 3      # cycles required to declare stagnation
    min_agents_for_intervention: int = 2  # don’t trigger on single-agent scenarios
    cooloff_seconds: float = 60.0         # avoid repeated rapid interventions

@dataclass
class MonitorIntervals:
    sample_seconds: float = 1.0   # system sample interval
    log_flush_every: int = 10     # write every N samples
    prometheus_port: Optional[int] = None  # e.g. 9108 to expose metrics

# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------
def _now_ts() -> float:
    return time.time()

def _ts_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _ewma(prev: float, new: float, alpha: float) -> float:
    return (alpha * new) + (1 - alpha) * prev

def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1

def _safe_write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Failed writing JSONL to %s", path)

# ------------------------------------------------------------------------------
# System resource monitor (process + system)
# ------------------------------------------------------------------------------
class SystemResourceMonitor:
    """
    Samples process/system CPU & memory, keeps rolling windows + EWMA.
    Thread-safe.
    """
    def __init__(self, window: int = 60, ewma_alpha: float = 0.2):
        self._proc = psutil.Process()
        self.window = window
        self.ewma_alpha = ewma_alpha
        self.process = psutil.Process()

        self._lock = threading.Lock()
        self.cpu_window: Deque[float] = deque(maxlen=window)
        self.mem_window: Deque[float] = deque(maxlen=window)
        self.sys_cpu_window: Deque[float] = deque(maxlen=window)
        self.sys_mem_window: Deque[float] = deque(maxlen=window)

        self.cpu_ewma: float = 0.0
        self.mem_ewma: float = 0.0
        self.sys_cpu_ewma: float = 0.0
        self.sys_mem_ewma: float = 0.0
        self.task_timings = defaultdict(lambda: deque(maxlen=20))
        self.cpu_history = deque(maxlen=20)
        self.mem_history = deque(maxlen=20)

    # existing methods…

    def get_cpu_usage(self) -> float:
        import psutil
        usage = self.process.cpu_percent(interval=0.1)
        self.cpu_history.append(usage)
        return usage

    def get_memory_usage(self) -> float:
        usage = self.process.memory_percent()
        self.mem_history.append(usage)
        return usage

    def sample(self) -> Dict[str, float]:
        # psutil quirk: cpu_percent works best if called with a small interval
        proc_cpu = self._proc.cpu_percent(interval=0.0)  # non-blocking since we sample externally
        proc_mem = self._proc.memory_percent()
        sys_cpu = psutil.cpu_percent(interval=None)
        sys_mem = psutil.virtual_memory().percent

        with self._lock:
            self.cpu_window.append(proc_cpu)
            self.mem_window.append(proc_mem)
            self.sys_cpu_window.append(sys_cpu)
            self.sys_mem_window.append(sys_mem)

            self.cpu_ewma = _ewma(self.cpu_ewma, proc_cpu, self.ewma_alpha)
            self.mem_ewma = _ewma(self.mem_ewma, proc_mem, self.ewma_alpha)
            self.sys_cpu_ewma = _ewma(self.sys_cpu_ewma, sys_cpu, self.ewma_alpha)
            self.sys_mem_ewma = _ewma(self.sys_mem_ewma, sys_mem, self.ewma_alpha)

            return {
                "proc_cpu": proc_cpu,
                "proc_mem": proc_mem,
                "sys_cpu": sys_cpu,
                "sys_mem": sys_mem,
                "proc_cpu_ewma": self.cpu_ewma,
                "proc_mem_ewma": self.mem_ewma,
                "sys_cpu_ewma": self.sys_cpu_ewma,
                "sys_mem_ewma": self.sys_mem_ewma,
            }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "proc_cpu_avg": mean(self.cpu_window) if self.cpu_window else 0.0,
                "proc_mem_avg": mean(self.mem_window) if self.mem_window else 0.0,
                "sys_cpu_avg": mean(self.sys_cpu_window) if self.sys_cpu_window else 0.0,
                "sys_mem_avg": mean(self.sys_mem_window) if self.sys_mem_window else 0.0,
                "proc_cpu_ewma": self.cpu_ewma,
                "proc_mem_ewma": self.mem_ewma,
                "sys_cpu_ewma": self.sys_cpu_ewma,
                "sys_mem_ewma": self.sys_mem_ewma,
            }

# ------------------------------------------------------------------------------
# Swarm health monitor (agents, tasks, stagnation, interventions)
# ------------------------------------------------------------------------------
class SwarmHealthMonitor:
    """
    Tracks agent heartbeats, intents, task durations (p99), detects stagnation,
    and can trigger interventions via tool_registry (e.g., shuffle roles).
    """
    def __init__(
        self,
        resource_thresholds: ResourceThresholds = ResourceThresholds(),
        stagnation_policy: StagnationPolicy = StagnationPolicy(),
        intervals: MonitorIntervals = MonitorIntervals(),
        log_path: str = SWARM_ACTIVITY_LOG,
        orchestrator_inject: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        tool_registry: Any = None,   # <-- NEW
    ):
        self.res_thresh = resource_thresholds
        self.stag = stagnation_policy
        self.intervals = intervals
        self.log_path = log_path
        self.inject_directives = orchestrator_inject  # e.g., orchestrator.inject_directives
        self.tool_registry = tool_registry            # <-- NEW

        # state
        self._lock = threading.Lock()
        self._agent_intents: Dict[str, str] = {}                   # agent -> current intent
        self._agent_last_seen: Dict[str, float] = {}               # agent -> ts
        self._task_latencies: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=200))
        self._idle_cycles: int = 0
        self._last_intervention_ts: float = 0.0

        # system monitor + background sampling
        self.sysmon = SystemResourceMonitor()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sample_count = 0

        # Optional Prometheus
        if _PROM_AVAILABLE and self.intervals.prometheus_port:
            start_http_server(self.intervals.prometheus_port)
            self._g_proc_cpu = Gauge("catalyst_proc_cpu_percent", "Process CPU %")
            self._g_proc_mem = Gauge("catalyst_proc_mem_percent", "Process Memory %")
            self._g_sys_cpu = Gauge("catalyst_sys_cpu_percent", "System CPU %")
            self._g_sys_mem = Gauge("catalyst_sys_mem_percent", "System Memory %")
        else:
            self._g_proc_cpu = self._g_proc_mem = self._g_sys_cpu = self._g_sys_mem = None

        logger.info("[SwarmHealthMonitor] initialized. JSONL=%s", self.log_path)

    # --------------------------------------------------------------------------
    # Back-compat & Orchestrator-facing health API
    # --------------------------------------------------------------------------
    def analyze_system_state(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Returns a health summary dict. No side effects by default.
        Pass perform_interventions=True to allow auto-intervention.
        """
        perform_interventions = bool(kwargs.get("perform_interventions", False))

        # Current system + agent snapshot
        sys_snap = self.sysmon.snapshot()
        with self._lock:
            num_agents = max(1, len(self._agent_last_seen))
            intents = list(self._agent_intents.values())
            idle_count = sum(1 for i in intents if i in self.stag.idle_intents)
            idle_ratio = idle_count / num_agents
            p99_over = self._p99_over_agents()

        alerts: List[str] = []
        status = "OK"

        if idle_ratio >= self.stag.idle_ratio_threshold and num_agents >= self.stag.min_agents_for_intervention:
            status = "IDLE"
            alerts.append(f"High idle ratio: {idle_ratio:.2f} (agents={num_agents})")

        if sys_snap["sys_cpu_ewma"] >= self.res_thresh.cpu_high_pct:
            status = "PRESSURE"
            alerts.append(f"High CPU pressure: {sys_snap['sys_cpu_ewma']:.1f}%")

        if sys_snap["sys_mem_ewma"] >= self.res_thresh.mem_high_pct:
            status = "PRESSURE"
            alerts.append(f"High MEM pressure: {sys_snap['sys_mem_ewma']:.1f}%")

        if p99_over:
            alerts.append(f"High task latency agents: {', '.join(p99_over)}")

        # optional side effect (kept off unless explicitly requested)
        if perform_interventions:
            try:
                self._detect_and_maybe_intervene()
            except Exception:
                logger.exception("intervention during analyze_system_state failed")

        return {
            "status": status,
            "idle_ratio": idle_ratio,
            "num_agents": num_agents,
            "p99_latency_over": p99_over,
            "system": sys_snap,
            "alerts": alerts,
        }

    # Aliases for older callers (safe no-ops if orchestrator expects these)
    def assess_system_health(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze_system_state(*args, **kwargs)

    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze_system_state(*args, **kwargs)

    def check(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze_system_state(*args, **kwargs)

    # --- public API: heartbeat / intents / timings ---------------------------------
    def heartbeat(self, agent_name: str, intent: Optional[str] = None) -> None:
        now = _now_ts()
        with self._lock:
            self._agent_last_seen[agent_name] = now
            if intent is not None:
                self._agent_intents[agent_name] = intent

    def record_task_start(self, agent_name: str) -> float:
        # return a token (start_ts) to feed into record_task_end
        return _now_ts()

    def record_task_end(self, agent_name: str, start_token: float) -> None:
        elapsed = max(0.0, _now_ts() - start_token)
        with self._lock:
            self._task_latencies[agent_name].append(elapsed)

    # --- background loop -------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="SwarmHealthMonitor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    # --- core logic -----------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            # sample system
            s = self.sysmon.sample()
            self._update_prom(s)

            # detect + possibly intervene
            try:
                self._detect_and_maybe_intervene()
            except Exception:
                logger.exception("detect/intervene failed")

            # emit snapshot periodically
            self._sample_count += 1
            if self._sample_count % self.intervals.log_flush_every == 0:
                self._emit_snapshot()

            # sleep for the remainder
            dt = time.time() - t0
            to_sleep = max(0.0, self.intervals.sample_seconds - dt)
            time.sleep(to_sleep)

    def _update_prom(self, s: Dict[str, float]) -> None:
        if not self._g_proc_cpu:
            return
        try:
            self._g_proc_cpu.set(s["proc_cpu"])
            self._g_proc_mem.set(s["proc_mem"])
            self._g_sys_cpu.set(s["sys_cpu"])
            self._g_sys_mem.set(s["sys_mem"])
        except Exception:
            pass

    def _emit_snapshot(self) -> None:
        with self._lock:
            agent_states = {a: self._agent_intents.get(a, "Unknown") for a in self._agent_last_seen.keys()}
            all_lat = []
            p99_per_agent: Dict[str, float] = {}
            for a, arr in self._task_latencies.items():
                if arr:
                    vals = sorted(arr)
                    p99 = _percentile(vals, 99)
                    p99_per_agent[a] = p99
                    all_lat.extend(arr)

            sys_snap = self.sysmon.snapshot()

        payload = {
            "timestamp": _ts_iso(),
            "event_type": "SWARM_SNAPSHOT",
            "source": "SwarmHealthMonitor",
            "description": "Periodic resource & agent snapshot",
            "details": {
                "agents": agent_states,
                "p99_latency_per_agent": p99_per_agent,
                "system": sys_snap,
            },
        }
        _safe_write_jsonl(self.log_path, payload)
        logger.debug("[Snapshot] %s", payload)

    # --- detection & interventions ---------------------------------------------------
    def _detect_and_maybe_intervene(self) -> None:
        with self._lock:
            intents = list(self._agent_intents.values())
            num_agents = max(1, len(self._agent_last_seen))
            idle_count = sum(1 for i in intents if i in self.stag.idle_intents)
            idle_ratio = idle_count / num_agents

        # stagnation cycles
        if idle_ratio >= self.stag.idle_ratio_threshold and num_agents >= self.stag.min_agents_for_intervention:
            self._idle_cycles += 1
            logger.info("[SwarmHealth] idle_ratio=%.2f cycles=%d", idle_ratio, self._idle_cycles)
        else:
            if self._idle_cycles:
                logger.info("[SwarmHealth] activity detected; reset idle cycle counter")
            self._idle_cycles = 0

        # system pressure check
        sys_snap = self.sysmon.snapshot()
        high_cpu = sys_snap["sys_cpu_ewma"] >= self.res_thresh.cpu_high_pct
        high_mem = sys_snap["sys_mem_ewma"] >= self.res_thresh.mem_high_pct

        # p99 latency
        p99_over = self._p99_over_agents()

        # decide interventions
        if self._idle_cycles >= self.stag.consecutive_idle_cycles:
            self._idle_cycles = 0
            self._do_intervention(kind="STAGNATION")

        if high_cpu or high_mem:
            reason = "CPU" if high_cpu else "MEM"
            self._emit_alert(f"System pressure high ({reason}). cpu={sys_snap['sys_cpu_ewma']:.1f} mem={sys_snap['sys_mem_ewma']:.1f}")

        if p99_over:
            self._emit_alert(f"High task latency agents detected: {', '.join(p99_over)}")

    def _p99_over_agents(self) -> List[str]:
        over = []
        with self._lock:
            for a, arr in self._task_latencies.items():
                if not arr:
                    continue
                vals = sorted(arr)
                p99 = _percentile(vals, 99)
                if p99 >= self.res_thresh.task_p99_latency_s:
                    over.append(a)
        return over

    def _emit_alert(self, msg: str) -> None:
        payload = {
            "timestamp": _ts_iso(),
            "event_type": "SWARM_ALERT",
            "source": "SwarmHealthMonitor",
            "description": msg,
            "details": {},
        }
        _safe_write_jsonl(self.log_path, payload)
        logger.warning("[ALERT] %s", msg)

    def _do_intervention(self, kind: str) -> None:
        # cooldown
        now = _now_ts()
        if now - self._last_intervention_ts < self.stag.cooloff_seconds:
            logger.info("[Intervention] skipped due to cooldown")
            return
        self._last_intervention_ts = now

        # pick stagnant agents for shuffle (those in idle intents)
        with self._lock:
            stagnant = [a for a, intent in self._agent_intents.items() if intent in self.stag.idle_intents]

        if not stagnant:
            logger.info("[Intervention] no stagnant agents identified")
            return

        # use tool_registry to generate exploratory tasks
        if not self.tool_registry or not hasattr(self.tool_registry, "safe_call"):
            self._emit_alert("Intervention tool unavailable (tool_registry.safe_call missing).")
            return

        result = self.tool_registry.safe_call("shuffle_roles_and_tasks", stagnant_agents=stagnant)

        directives: List[Dict[str, Any]] = []
        if isinstance(result, list):
            directives = result
        else:
            # tool returned an error string? log it and bail
            self._emit_alert(f"Intervention tool failed: {result}")
            return

        # emit event
        payload = {
            "timestamp": _ts_iso(),
            "event_type": "INTERVENTION",
            "source": "SwarmHealthMonitor",
            "description": f"{kind} detected; injecting exploratory tasks",
            "details": {"stagnant_agents": stagnant, "directives": directives},
        }
        _safe_write_jsonl(self.log_path, payload)
        logger.info("[Intervention] %s -> %d directives", kind, len(directives))

        # push to orchestrator if provided
        if self.inject_directives and directives:
            try:
                self.inject_directives(directives)
            except Exception:
                logger.exception("Failed injecting directives during intervention")

# ------------------------------------------------------------------------------
# Log tailer (safe and patient)
# ------------------------------------------------------------------------------
def tail_log(filepath: str) -> None:
    """
    Simple JSONL tailer for cockpit/debug. Waits for file creation if absent.
    """
    print(f"--- Swarm Monitor (Tailing: {filepath}) ---")

    while not os.path.exists(filepath):
        print(f"[Tail] Waiting for log file to appear: {filepath}")
        time.sleep(1)

    with open(filepath, "r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            try:
                entry = json.loads(line.strip())
                ts = entry.get("timestamp", "N/A")
                evt = entry.get("event_type", "N/A")
                src = entry.get("source", "N/A")
                desc = entry.get("description", "N/A")
                details = entry.get("details", {})
                details_str = json.dumps(details, indent=2) if details else "{}"
                print(f"[{ts}] <{evt}> Source: {src}\n  {desc}\n  Details: {details_str}\n---")
            except Exception:
                print(f"[Tail] Bad JSON line: {line.strip()}")

# ------------------------------------------------------------------------------
# Optional: minimal self-test / CLI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Quick local smoke test:
      python -m swarm_monitor
    Then in another shell: `tail -f persistence_data/swarm_activity.jsonl`
    """
    logging.basicConfig(level=logging.INFO)

    mon = SwarmHealthMonitor(
        intervals=MonitorIntervals(sample_seconds=0.5, log_flush_every=4, prometheus_port=None),
    )
    mon.start()

    agents = ["ProtoAgent_Observer_instance_1", "ProtoAgent_Worker_instance_1", "ProtoAgent_Security_instance_1"]

    # simulate heartbeats + idle
    try:
        t0 = time.time()
        while time.time() - t0 < 15:
            for a in agents:
                mon.heartbeat(a, intent="Awaiting tasks from the Planner.")
            # simulate a task latency
            st = mon.record_task_start(agents[1])
            time.sleep(0.2)
            mon.record_task_end(agents[1], st)
            time.sleep(0.5)
        print("Stopping monitor...")
    finally:
        mon.stop()
