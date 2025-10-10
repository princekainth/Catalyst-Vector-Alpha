# tools.py — clean, circular-import-safe tool implementations + lazy registration

from __future__ import annotations

import os
import re
import time
import json
import math
import psutil
import random
import shutil
import logging
import hashlib
import subprocess
import shlex
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from ipaddress import ip_address

# Optional deps (guarded)
try:
    import requests
except Exception:
    requests = None  # handled in code

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None  # handled in code

try:
    from fpdf import FPDF
except Exception:
    FPDF = None

try:
    from transformers import pipeline as _hf_pipeline
except Exception:
    _hf_pipeline = None


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger("CatalystLogger")


# ------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------
_PLACEHOLDERS = {"", " ", "string", "placeholder", "tbd", "todo", "none", "null", "n/a", "na", "<placeholder>"}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and v.strip().lower() in _PLACEHOLDERS

def _require_non_placeholder(name: str, value: Any) -> Optional[str]:
    return None if not _is_placeholder(value) else f"'{name}' is missing or placeholder."

def _valid_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.scheme in {"http", "https"} and bool(u.netloc)
    except Exception:
        return False

def standardize_response(status: str, data: Any = None, error: str | None = None, **meta) -> dict:
    """Consistent response wrapper across tools."""
    res = {"status": status, "timestamp": _now_iso()}
    if data is not None:
        res["data"] = data
    if error:
        res["error"] = error
    if meta:
        res.update(meta)
    return res


# ------------------------------------------------------------------------------
# Validators (used during registration by the central registry)
# ------------------------------------------------------------------------------
def _v_url(field: str) -> Callable[[Dict[str, Any]], Optional[str]]:
    def _v(args: Dict[str, Any]) -> Optional[str]:
        val = (args.get(field) or "").strip()
        return None if _valid_url(val) else f"'{field}' must be a valid http(s) URL."
    return _v

def _v_enum(field: str, allowed: set[str]) -> Callable[[Dict[str, Any]], Optional[str]]:
    allowed_l = {a.lower() for a in allowed}
    def _v(args: Dict[str, Any]) -> Optional[str]:
        v = (args.get(field) or "").strip().lower()
        return None if v in allowed_l else f"'{field}' must be one of {sorted(list(allowed))}."
    return _v

def _v_ipv4(field: str) -> Callable[[Dict[str, Any]], Optional[str]]:
    def _v(args: Dict[str, Any]) -> Optional[str]:
        val = args.get(field)
        if val is None:
            return f"Missing required field '{field}'."
        try:
            ip = ip_address(val)
            if ip.version != 4:
                return f"'{field}' must be an IPv4 address."
        except Exception:
            return f"Invalid IPv4 address for '{field}'."
        return None
    return _v

def _v_has_namespace(args: dict) -> Optional[str]:
    ns = (args.get("namespace") or "").strip()
    return None if ns and not _is_placeholder(ns) else "'namespace' is required"

def _v_k8s_scale_args(args: dict) -> Optional[str]:
    # must have deployment or alias 'name' and replicas>=1
    name = args.get("deployment") or args.get("name")
    if not (isinstance(name, str) and name.strip() and not _is_placeholder(name)):
        return "either 'deployment' or 'name' is required"
    try:
        r = int(args.get("replicas"))
        if r < 1:
            return "'replicas' must be >= 1"
    except Exception:
        return "'replicas' must be an integer"
    return None


# ------------------------------------------------------------------------------
# Prometheus helpers
# ------------------------------------------------------------------------------
def _prom_url() -> Optional[str]:
    return os.getenv("PROMETHEUS_URL")

def _prom_request(path: str, params: Dict[str, Any], timeout: float = 10.0) -> dict:
    if requests is None:
        return standardize_response("error", error="python-requests not installed")
    base = _prom_url()
    if not base:
        return standardize_response("error", error="PROMETHEUS_URL not set")
    try:
        r = requests.get(f"{base.rstrip('/')}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return standardize_response("ok", data=r.json())
    except Exception as e:
        return standardize_response("error", error=str(e))


# ------------------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------------------

# ---- System / Local ----------------------------------------------------------
def get_system_cpu_load_tool(time_interval_seconds: float = 0.5, samples: int = 3, per_core: bool = False) -> dict:
    # clamp
    interval = max(0.0, min(float(time_interval_seconds), 5.0))
    samples = max(1, min(int(samples), 5))
    try:
        readings: List[Any] = []
        for _ in range(samples):
            readings.append(psutil.cpu_percent(interval=interval, percpu=per_core))
        if per_core:
            cores = len(readings[0]) if readings else 0
            averaged = [round(sum(s[i] for s in readings)/len(readings), 2) for i in range(cores)]
            data = averaged
            summary = f"Per-core CPU: {averaged}"
        else:
            avg = sum(readings) / len(readings)
            data = round(float(avg), 2)
            summary = f"CPU load: {data}%"
        return standardize_response("ok", data=data, summary=summary, unit="percent")
    except Exception as e:
        return standardize_response("error", error=str(e), summary="Failed to get CPU load")

def get_system_resource_usage_tool() -> dict:
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory().percent
        return standardize_response("ok", data={"cpu_percent": cpu, "memory_percent": mem},
                                    summary=f"CPU {cpu}%, MEM {mem}%")
    except Exception as e:
        return standardize_response("error", error=str(e), summary="resource usage failure")

def disk_usage_tool(path: str = "/") -> dict:
    try:
        u = psutil.disk_usage(path)
        return standardize_response(
            "ok",
            data={"path": path, "total_bytes": u.total, "used_bytes": u.used, "free_bytes": u.free, "percent": u.percent},
            summary=f"{path}: {u.percent}% used",
        )
    except Exception as e:
        return standardize_response("error", error=f"disk_usage_tool failed: {e}", data={"path": path})

def top_processes_tool(limit: int = 10) -> dict:
    """Return top processes by CPU% (dict form)."""
    try:
        # warm-up for accurate cpu_percent
        procs = [p for p in psutil.process_iter(attrs=["pid", "name"])]
        for p in procs:
            try:
                p.cpu_percent(None)
            except Exception:
                pass
        time.sleep(0.25)
        rows = []
        for p in procs:
            try:
                rows.append({
                    "pid": p.pid,
                    "name": p.info.get("name"),
                    "cpu_percent": p.cpu_percent(None),
                    "memory_percent": p.memory_percent(),
                })
            except Exception:
                continue
        rows.sort(key=lambda r: r.get("cpu_percent") or 0.0, reverse=True)
        rows = rows[: max(1, min(int(limit), 50))]
        return standardize_response("ok", data={"processes": rows, "count": len(rows)},
                                    summary=f"Top {len(rows)} processes by CPU")
    except Exception as e:
        return standardize_response("error", error=str(e))

def measure_responsiveness_tool(**kwargs) -> dict:
    """Rough 'open time' by timing a tiny Python run (compatible signature)."""
    try:
        start = time.time()
        subprocess.run(["python3", "-c", "print(1)"], capture_output=True, timeout=2)
        elapsed_ms = (time.time() - start) * 1000.0
        return standardize_response("ok", data={"open_time_ms": round(elapsed_ms, 2), "responsive": elapsed_ms < 500})
    except Exception as e:
        return standardize_response("error", error=str(e))


# ---- Kubernetes --------------------------------------------------------------
def _parse_kubectl_top_pods(raw: str) -> List[Dict[str, Any]]:
    """
    Parse `kubectl top pods -A --no-headers` lines:
    NAMESPACE NAME CPU(cores) MEMORY(bytes)
    """
    rows: List[Dict[str, Any]] = []
    for line in filter(None, (l.strip() for l in raw.splitlines())):
        parts = line.split()
        if len(parts) < 4:
            continue
        ns, name, cpu_s, mem_s = parts[0], parts[1], parts[2], parts[3]

        def cpu_to_mcores(v: str) -> Optional[int]:
            try:
                return int(v[:-1]) if v.endswith("m") else int(float(v) * 1000)
            except Exception:
                return None

        def mem_to_Mi(v: str) -> Optional[float]:
            try:
                if v.endswith("Mi"):  return float(v[:-2])
                if v.endswith("Gi"):  return float(v[:-2]) * 1024.0
                if v.endswith("Ki"):  return float(v[:-2]) / 1024.0
                # assume bytes
                return float(v) / (1024.0 * 1024.0)
            except Exception:
                return None

        rows.append({
            "namespace": ns,
            "pod": name,
            "cpu_mcores": cpu_to_mcores(cpu_s),
            "memory_Mi": mem_to_Mi(mem_s),
            "raw_cpu": cpu_s,
            "raw_memory": mem_s,
        })
    return rows

def kubernetes_pod_metrics_tool(namespace: Optional[str] = None,
                                selector: Optional[str] = None,
                                limit: int = 50) -> dict:
    """Use `kubectl top pods` for real pod metrics (needs metrics-server)."""
    if not shutil.which("kubectl"):
        return standardize_response("error", error="kubectl not found on PATH",
                                    cmd="kubectl top pods -A --no-headers")
    cmd = ["kubectl", "top", "pods"]
    if namespace:
        cmd += ["-n", namespace]
    else:
        cmd += ["-A"]
    cmd += ["--no-headers"]
    if selector:
        cmd += ["-l", selector]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=8)
        rows = _parse_kubectl_top_pods(out)
        if limit:
            rows = rows[: max(1, int(limit))]
        total_cpu = sum(r.get("cpu_mcores") or 0 for r in rows)
        total_mem = sum(r.get("memory_Mi") or 0.0 for r in rows)
        return standardize_response(
            "ok",
            data={"pods": rows, "count": len(rows), "total_cpu_mcores": total_cpu, "total_memory_Mi": round(total_mem, 2)},
            cmd=" ".join(shlex.quote(x) for x in cmd),
            summary=f"{len(rows)} pods, total CPU {total_cpu}m",
        )
    except subprocess.TimeoutExpired:
        return standardize_response("error", error="kubectl top timed out", cmd=" ".join(cmd))
    except subprocess.CalledProcessError as e:
        return standardize_response("error", error=f"kubectl failed: {e}", cmd=" ".join(cmd))

_SCALE_MIN_INTERVAL_S = float(os.getenv("CVA_SCALE_MIN_INTERVAL_S", "300"))  # rate limit per deployment

def _kubectl_json(cmd: List[str]) -> Any:
    out = subprocess.check_output(cmd, text=True, timeout=8)
    return json.loads(out)

def _get_deploy(ns: str, name: str) -> Dict[str, Any]:
    return _kubectl_json(["kubectl", "-n", ns, "get", "deploy", name, "-o", "json"])

def k8s_scale_tool(namespace: str,
                   deployment: str,
                   replicas: int,
                   dry_run: bool = True,
                   approval_token: Optional[str] = None,
                   min_replicas: int = 1,
                   max_replicas: int = 10) -> dict:
    """Safe, subprocess-based scaler with guardrails and rate limiting."""
    # sanitize
    try:
        replicas = int(replicas)
        min_replicas = int(min_replicas)
        max_replicas = int(max_replicas)
    except Exception:
        return {"ok": False, "error": "replicas/min/max must be integers"}
    if min_replicas < 0 or max_replicas < 0 or min_replicas > max_replicas:
        return {"ok": False, "error": "invalid min/max replicas bounds"}
    target = max(min_replicas, min(max_replicas, replicas))

    # rate limit per (ns, deployment)
    key_path = f"/tmp/cva_scale_{namespace}_{deployment}.ts"
    try:
        with open(key_path, "r") as f:
            last_ts = float(f.read().strip() or "0")
    except Exception:
        last_ts = 0.0
    now = time.time()
    if now - last_ts < _SCALE_MIN_INTERVAL_S:
        return {"ok": False, "reason": "rate_limited", "retry_in_s": round(_SCALE_MIN_INTERVAL_S - (now - last_ts), 1)}

    # fetch current replicas
    try:
        dep = _get_deploy(namespace, deployment)
        current = int(dep["spec"].get("replicas", 1))
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"kubectl get failed: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"could not read deployment: {e}"}

    plan = {
        "namespace": namespace,
        "deployment": deployment,
        "current": current,
        "target": target,
        "min": min_replicas,
        "max": max_replicas,
        "dry_run": bool(dry_run),
    }
    if dry_run or not approval_token:
        plan["action"] = "plan"
        plan["note"] = "dry-run or missing approval_token"
        return {"ok": True, **plan}

    # execute
    cmd = ["kubectl", "-n", namespace, "scale", "deploy", deployment, f"--replicas={target}"]
    try:
        subprocess.check_call(cmd, timeout=8)
        try:
            with open(key_path, "w") as f:
                f.write(str(now))
        except Exception:
            pass
        rollback = {"namespace": namespace, "deployment": deployment, "prev_replicas": current, "ts": now}
        return {"ok": True, "action": "scaled", "rolled_from": current, "rolled_to": target,
                "cmd": " ".join(shlex.quote(x) for x in cmd), "rollback": rollback}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "kubectl scale timed out", "cmd": " ".join(shlex.quote(x) for x in cmd)}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"kubectl scale failed: {e}", "cmd": " ".join(shlex.quote(x) for x in cmd)}
    except Exception as e:
        return {"ok": False, "error": f"unexpected: {e}", "cmd": " ".join(shlex.quote(x) for x in cmd)}


# ---- Security / Networking ---------------------------------------------------
_ALLOWED_SCAN_TYPES = {"ping_sweep", "full_port_scan", "vulnerability_scan"}

def initiate_network_scan_tool(target_ip: str, scan_type: str = "ping_sweep") -> str:
    err = _require_non_placeholder("target_ip", target_ip)
    if err:
        return f"[ERROR] initiate_network_scan_tool: {err}"
    scan_type = (scan_type or "ping_sweep").strip().lower()
    if scan_type not in _ALLOWED_SCAN_TYPES:
        logger.warning(f"[TOOL EXEC] Unknown scan_type '{scan_type}', defaulting to ping_sweep.")
        scan_type = "ping_sweep"
    logger.info(f"[TOOL EXEC] network_scan: {scan_type} on {target_ip}")
    time.sleep(0.2)
    if scan_type == "full_port_scan":
        open_ports = random.sample([21, 22, 23, 80, 443, 3389, 8080], k=random.randint(1, 3))
        return f"Port scan on {target_ip} complete. Open ports: {open_ports}"
    if scan_type == "vulnerability_scan":
        if random.random() < 0.1:
            vuln = random.choice(["CVE-2023-1234 (High)", "CVE-2022-5678 (Medium)"])
            return f"Vulnerability scan found: {vuln}"
        return f"No critical vulnerabilities found on {target_ip}"
    return f"Successfully pinged {target_ip}. Host is up."

def deploy_recovery_protocol_tool(protocol_name: str, target_system_id: str, urgency_level: str = "medium") -> str:
    for k, v in (("protocol_name", protocol_name), ("target_system_id", target_system_id)):
        err = _require_non_placeholder(k, v)
        if err:
            return f"[ERROR] deploy_recovery_protocol_tool: {err}"
    urgency_level = (urgency_level or "medium").strip().lower()
    if urgency_level not in {"low", "medium", "high", "critical"}:
        logger.warning(f"[TOOL EXEC] Unknown urgency_level '{urgency_level}', defaulting to 'medium'.")
        urgency_level = "medium"
    logger.info(f"[TOOL EXEC] deploy_recovery: {protocol_name} -> {target_system_id} ({urgency_level})")
    time.sleep(0.2)
    return f"Recovery protocol '{protocol_name}' deployed to {target_system_id} (Urgency: {urgency_level})."

def analyze_threat_signature_tool(signature: str, source_ip: str) -> dict:
    for k, v in (("signature", signature), ("source_ip", source_ip)):
        err = _require_non_placeholder(k, v)
        if err:
            return standardize_response("error", error=err)
    risk = random.choice(["Low", "Medium", "High", "Critical"])
    return standardize_response("ok",
                               data={"signature": signature, "source_ip": source_ip, "risk_level": risk},
                               summary=f"Analysis of {signature} from {source_ip}: Risk={risk}")

def isolate_network_segment_tool(segment_id: str, reason: str) -> dict:
    for k, v in (("segment_id", segment_id), ("reason", reason)):
        err = _require_non_placeholder(k, v)
        if err:
            return standardize_response("error", error=f"isolate_network_segment_tool: {err}",
                                        summary="Isolation failed (bad args)")
    return standardize_response("ok", data={"segment_id": segment_id, "reason": reason},
                                summary=f"Segment '{segment_id}' isolated")

def extract_iocs_tool(text: str) -> dict:
    if _is_placeholder(text):
        return standardize_response("error", error="Text is empty/placeholder.", summary="IOC extraction failed")
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    urls = re.findall(r"https?://[^\s)]+", text)
    sha256 = re.findall(r"\b[a-fA-F0-9]{64}\b", text)
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    data = {"ips": sorted(set(ips)), "urls": sorted(set(urls)), "sha256": sorted(set(sha256)), "emails": sorted(set(emails))}
    total = sum(len(v) for v in data.values())
    return standardize_response("ok", data=data, summary=f"Extracted {total} IOCs")

def hash_text_tool(text: str, algorithm: str = "sha256") -> dict:
    if _is_placeholder(text):
        return standardize_response("error", error="Text is empty/placeholder.", summary="Hash failed")
    algo = (algorithm or "sha256").lower()
    try:
        h = hashlib.new(algo)
    except Exception:
        return standardize_response("error", error=f"Unsupported hash algorithm '{algorithm}'.", summary="Hash failed")
    h.update(text.encode("utf-8", errors="ignore"))
    return standardize_response("ok", data={"algorithm": algo, "hexdigest": h.hexdigest()},
                                summary=f"Hashed text with {algo}")


# ---- Environment / World / Knowledge ----------------------------------------
def get_environmental_data_tool(location: Optional[str] = "server_room_3",
                                data_type: str = "all",
                                use_real_sensors: bool = False) -> dict:
    """
    If use_real_sensors=True and SENSOR_API_URL is set, tries GET {SENSOR_API_URL}/sensors/{location}.
    Otherwise returns synthetic readings.
    """
    # optional real sensor
    if use_real_sensors and requests is not None:
        try:
            base = os.getenv("SENSOR_API_URL")
            if base and location:
                r = requests.get(f"{base.rstrip('/')}/sensors/{location}", timeout=5)
                if r.ok:
                    return standardize_response("ok", data=r.json(), location=location, source="sensor_api")
        except Exception as e:
            logger.warning(f"Sensor API failed, fallback to mock: {e}")

    # synthetic fallback
    reading = {
        "temperature_celsius": round(19.5 + random.random() * 6.0, 2),
        "humidity_percent": round(30 + random.random() * 25, 1),
        "air_quality_index": int(40 + random.random() * 40),
    }
    allowed = {"all", "temperature_celsius", "humidity_percent", "air_quality_index"}
    if data_type not in allowed:
        return standardize_response("error", error=f"unsupported data_type: {data_type}")
    payload = reading if data_type == "all" else {data_type: reading[data_type]}
    return standardize_response("ok", data=payload, location=location, data_type=data_type, source="simulated")

_SERP_TIMEOUT = 12
def web_search_tool(query: str) -> dict:
    logger.info(f"[TOOL EXEC] web_search: {query}")
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return standardize_response("error", error="Missing SERPAPI_API_KEY environment variable.",
                                    summary="Search failed (no API key)")
    try:
        from serpapi import GoogleSearch  # lazy import
        params = {"q": query, "api_key": api_key, "engine": "google"}
        results = GoogleSearch(params).get_dict().get("organic_results", [])
        links = [{"title": r.get("title"), "link": r.get("link")} for r in results[:5]]
        return standardize_response("ok", data={"results": links}, summary=f"Top {len(links)} results for '{query}'")
    except Exception as e:
        return standardize_response("error", error=f"web_search_tool failed: {e}", summary="Search failed")

_READ_CACHE_TTL = 600
_read_cache: Dict[str, Tuple[float, str]] = {}
def read_webpage_tool(url: str) -> str:
    url = (url or "").strip()
    if not url or not _valid_url(url):
        return "[ERROR] read_webpage_tool: 'url' must be a valid http(s) URL."
    if requests is None:
        return "[ERROR] read_webpage_tool: python-requests not installed."
    try:
        ts_val = _read_cache.get(url)
        if ts_val and time.time() - ts_val[0] < _READ_CACHE_TTL:
            return ts_val[1]
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        if BeautifulSoup is None:
            # fallback to raw text
            txt = resp.text[:4000]
            _read_cache[url] = (time.time(), txt)
            return txt
        soup = BeautifulSoup(resp.text, "lxml") if BeautifulSoup else None
        if soup is None:
            txt = resp.text[:4000]
            _read_cache[url] = (time.time(), txt)
            return txt
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        text = " ".join(soup.get_text().split())
        snippet = text[:4000] or "[WARN] Page has no readable text."
        _read_cache[url] = (time.time(), snippet)
        return snippet
    except Exception as e:
        return f"[ERROR] read_webpage_tool failed: {e}"

def update_world_model_tool(key: str, value: str) -> dict:
    err = _require_non_placeholder("key", key) or _require_non_placeholder("value", value)
    if err:
        return standardize_response("error", error=f"update_world_model_tool: {err}", summary="World model update failed")
    return standardize_response("ok", data={"key": key, "value": value},
                                summary=f"World model updated: {key}={value}")

def query_long_term_memory_tool(query_text: str) -> dict:
    err = _require_non_placeholder("query_text", query_text)
    if err:
        return standardize_response("error", error=f"query_long_term_memory_tool: {err}", summary="LTM query failed")
    # hook your vector search here
    return standardize_response("ok", data={"query": query_text}, summary=f"LTM queried: '{query_text}'")


# ---- Text / ML / Reporting ---------------------------------------------------
_sentiment = None
def _get_sentiment_pipeline():
    global _sentiment
    if _sentiment is None and _hf_pipeline is not None:
        try:
            _sentiment = _hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            logger.error(f"Failed to load sentiment pipeline: {e}")
            _sentiment = None
    return _sentiment

def analyze_text_sentiment_tool(text: str) -> dict:
    if _is_placeholder(text):
        return {"error": "Text is empty/placeholder."}
    p = _get_sentiment_pipeline()
    if not p:
        return {"error": "Sentiment model unavailable."}
    try:
        return p(text)[0]
    except Exception as e:
        return {"error": f"Sentiment analysis failed: {e}"}

def redact_pii_tool(text: str) -> dict:
    if _is_placeholder(text):
        return {"error": "Text is empty/placeholder."}
    red = text
    red = re.sub(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", r"***@\2", red)
    red = re.sub(r"\b(\+?\d[\d\s\-]{7,}\d)\b", "[REDACTED-PHONE]", red)
    return {"redacted_text": red}

def _safe_pdf_name(name: str) -> str:
    stem = Path(name).stem or f"report-{int(time.time())}"
    safe = "".join(c for c in stem if c.isalnum() or c in ("_", "-")) or "report"
    return f"{safe}.pdf"

def create_pdf_tool(filename: str, text_content: str) -> str:
    os.makedirs("output", exist_ok=True)
    if FPDF is None:
        return "[ERROR] create_pdf_tool failed: FPDF not available."
    path = os.path.join("output", _safe_pdf_name(filename))
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=text_content.encode("latin-1", "replace").decode("latin-1"))
        pdf.output(path)
        return f"PDF saved: {path}"
    except Exception as e:
        return f"[ERROR] create_pdf_tool failed: {e}"

def generate_report_pdf_tool(title: str, sections: List[Dict[str, str]], filename: Optional[str] = None) -> str:
    if not sections or not isinstance(sections, list):
        return "[ERROR] generate_report_pdf_tool: 'sections' must be a non-empty list of {heading, body}."
    if not filename:
        filename = f"{title or 'Report'}-{int(time.time())}.pdf"
    lines = [f"# {title or 'Report'}", ""]
    for sec in sections:
        h = sec.get("heading") or "Section"
        b = sec.get("body") or ""
        lines.append(f"\n## {h}\n\n{b}\n")
    return create_pdf_tool(filename=filename, text_content="\n".join(lines))

def shuffle_roles_and_tasks_tool(stagnant_agents: List[str]) -> List[dict]:
    pool = {
        "observer": [
            "Cross-check system metrics against public datasets.",
            "Perform forensic analysis of last 100 events.",
            "Predict CPU load trends.",
        ],
        "security": [
            "Proactively hunt for emerging CVEs.",
            "Design heuristic for anomalous logins.",
            "Simulate red-team probe on sandbox network.",
        ],
        "collector": [
            "Integrate novel data source from open web.",
            "Audit integrity of all active data streams.",
            "Summarize past 24h of ingested data.",
        ],
    }
    directives = []
    for agent in stagnant_agents or []:
        role = "observer"
        if isinstance(agent, str) and "Security" in agent:
            role = "security"
        elif isinstance(agent, str) and "Collector" in agent:
            role = "collector"
        task = random.choice(pool[role])
        directives.append({"type": "AGENT_PERFORM_TASK", "agent_name": agent, "task_description": task})
    return directives

# --- Resource tuning ---------------------------------------------------------
def update_resource_allocation_tool(
    resource_type: str,
    target_agent_name: str,
    new_allocation_percentage,
    **kwargs
):
    """
    Idempotent no-op/stub that normalizes and returns the requested allocation.
    Accepts values like 0.35 (35%) or 35 (percent). Bounds to [0, 100].
    """
    try:
        val = float(new_allocation_percentage)
    except Exception:
        return {"status": "error", "error": "new_allocation_percentage must be a number"}

    # Normalize: if it's 0..1 treat as fraction; if >1 treat as percent
    if 0.0 <= val <= 1.0:
        pct = val * 100.0
    else:
        pct = val

    # Clamp just to be safe
    pct = max(0.0, min(100.0, pct))

    if not isinstance(resource_type, str) or not resource_type.strip():
        return {"status": "error", "error": "resource_type is required"}
    if not isinstance(target_agent_name, str) or not target_agent_name.strip():
        return {"status": "error", "error": "target_agent_name is required"}

    # This stub doesn’t actually change resources; it just echoes a deterministic result.
    # If you later wire real behavior, keep the same return shape.
    return {
        "status": "ok",
        "action": "update_resource_allocation",
        "resource_type": resource_type.strip().lower(),
        "target_agent_name": target_agent_name.strip(),
        "requested_percent": pct,
        "applied_percent": pct,   # in a real impl, reflect any policy clamps here
        "idempotent": True,
    }


# ---- Prometheus query tools --------------------------------------------------
def prometheus_query_tool(query: str, timeout_s: int = 10) -> dict:
    if not query or not isinstance(query, str):
        return standardize_response("error", error="query is required")
    res = _prom_request("/api/v1/query", {"query": query}, timeout=float(timeout_s))
    if res.get("status") != "ok":
        return res
    data = res["data"]
    # convenience summary
    try:
        result = data.get("data", {}).get("result", [])
        summary = {"result_type": data.get("data", {}).get("resultType"), "series": len(result)}
    except Exception:
        summary = {}
    return {"status": "ok", "data": data, "meta": {"summary": summary}, "ts": time.time()}

def prometheus_range_query_tool(query: str, start_s: int, end_s: int, step_s: int = 15, timeout_s: int = 10) -> dict:
    if not query or not isinstance(query, str):
        return standardize_response("error", error="query is required")
    if not all(isinstance(x, (int, float)) for x in (start_s, end_s, step_s)):
        return standardize_response("error", error="start_s, end_s, step_s must be numbers (unix seconds)")
    if end_s <= start_s:
        return standardize_response("error", error="end_s must be > start_s")
    res = _prom_request("/api/v1/query_range",
                        {"query": query, "start": int(start_s), "end": int(end_s), "step": int(step_s)},
                        timeout=float(timeout_s))
    if res.get("status") != "ok":
        return res
    data = res["data"]
    try:
        result = data.get("data", {}).get("result", [])
        summary = {
            "result_type": data.get("data", {}).get("resultType"),
            "series": len(result),
            "points_per_series": (len(result[0]["values"]) if result else 0),
        }
    except Exception:
        summary = {}
    return {"status": "ok", "data": data, "meta": {"summary": summary}, "ts": time.time()}


# ------------------------------------------------------------------------------
# Lazy registration (call ONCE from tool_registry.py or app.py)
# ------------------------------------------------------------------------------
def register_tools_into(registry) -> None:
    """
    Register all tools into the central ToolRegistry instance.
    Import ToolSpec *inside* to avoid circular imports.
    """
    from tool_registry import ToolSpec  # lazy import to avoid cycles

    # Observability / Prometheus
    registry.register_tool("prometheus_query", ToolSpec(
        fn=prometheus_query_tool,
        task_type="Observation",
        roles_allowed={"Observer", "Planner", "Worker"},
        timeout_s=15, retries=1, idempotent=True,
        required_args={"query"}, cache_ttl_s=1,
    ))
    registry.register_tool("prometheus_range_query", ToolSpec(
        fn=prometheus_range_query_tool,
        task_type="Observation",
        roles_allowed={"Observer", "Planner"},
        timeout_s=20, retries=1, idempotent=True,
        required_args={"query", "start_s", "end_s"},
    ))

    # System / Local
    registry.register_tool("get_system_cpu_load", ToolSpec(
        fn=get_system_cpu_load_tool,
        task_type="Observation",
        roles_allowed={"Observer", "Worker", "Security"},
        timeout_s=5, idempotent=True,
    ))
    registry.register_tool("get_system_resource_usage", ToolSpec(
        fn=get_system_resource_usage_tool,
        task_type="Observation",
        roles_allowed={"Observer", "Worker", "Security"},
        timeout_s=5, idempotent=True,
    ))
    registry.register_tool("disk_usage", ToolSpec(
        fn=disk_usage_tool,
        task_type="Observation",
        roles_allowed={"Observer", "Worker", "Security"},
        timeout_s=8, idempotent=True,
    ))
    registry.register_tool("top_processes", ToolSpec(
        fn=top_processes_tool,
        task_type="Observation",
        roles_allowed={"Observer", "Security"},
        timeout_s=10, idempotent=True,
    ))
    registry.register_tool("measure_responsiveness", ToolSpec(
        fn=measure_responsiveness_tool,
        task_type="Observation",
        roles_allowed={"Observer", "Security"},
        timeout_s=5, idempotent=True,
    ))

    # Kubernetes
    registry.register_tool("kubernetes_pod_metrics", ToolSpec(
        fn=kubernetes_pod_metrics_tool,
        task_type="Observation",
        roles_allowed={"Observer"},
        timeout_s=8, idempotent=True,
    ))
    registry.register_tool("k8s_scale", ToolSpec(
        fn=k8s_scale_tool,
        task_type="Actuation",
        roles_allowed={"Worker"},
        timeout_s=12, retries=0, idempotent=False,
        arg_aliases={"name": "deployment"},
        required_args={"namespace", "replicas"},
        validators=[_v_has_namespace, _v_k8s_scale_args],
    ))

    # Security / Net
    registry.register_tool("initiate_network_scan", ToolSpec(
        fn=initiate_network_scan_tool,
        task_type="GenericTask",
        roles_allowed={"Security", "Observer"},
        timeout_s=15, retries=1, idempotent=False,
        required_args={"target_ip"},
    ))
    registry.register_tool("deploy_recovery_protocol", ToolSpec(
        fn=deploy_recovery_protocol_tool,
        task_type="GenericTask",
        roles_allowed={"Security", "Worker"},
        timeout_s=20, retries=1, idempotent=False,
        required_args={"protocol_name", "target_system_id"},
        validators=[_v_enum("urgency_level", {"low", "medium", "high", "critical"})],
    ))
    registry.register_tool("analyze_threat_signature", ToolSpec(
        fn=analyze_threat_signature_tool,
        task_type="GenericTask",
        roles_allowed={"Security"},
        timeout_s=10, idempotent=True,
        required_args={"signature", "source_ip"},
        validators=[_v_ipv4("source_ip")],
    ))
    registry.register_tool("isolate_network_segment", ToolSpec(
        fn=isolate_network_segment_tool,
        task_type="GenericTask",
        roles_allowed={"Security"},
        timeout_s=10, idempotent=False,
        required_args={"segment_id", "reason"},
    ))
    registry.register_tool("extract_iocs", ToolSpec(
        fn=extract_iocs_tool,
        task_type="GenericTask",
        roles_allowed={"Security", "Observer", "Worker", "Planner"},
        timeout_s=6, idempotent=True,
        required_args={"text"},
    ))
    registry.register_tool("hash_text", ToolSpec(
        fn=hash_text_tool,
        task_type="GenericTask",
        roles_allowed={"Observer", "Security", "Worker", "Planner"},
        timeout_s=5, idempotent=True,
        required_args={"text"},
        validators=[_v_enum("algorithm", {"md5", "sha1", "sha224", "sha256", "sha384", "sha512"})],
    ))

    # Env / World / Knowledge
    registry.register_tool("get_environmental_data", ToolSpec(
        fn=get_environmental_data_tool,
        task_type="GenericTask",
        roles_allowed={"Observer", "Worker"},
        timeout_s=8, idempotent=True,
    ))
    registry.register_tool("web_search", ToolSpec(
        fn=web_search_tool,
        task_type="GenericTask",
        roles_allowed={"Planner", "Observer", "Security"},
        timeout_s=_SERP_TIMEOUT, retries=1, idempotent=True,
        required_args={"query"},
    ))
    registry.register_tool("read_webpage", ToolSpec(
        fn=read_webpage_tool,
        task_type="GenericTask",
        roles_allowed={"Planner", "Observer"},
        timeout_s=12, retries=1, idempotent=True,
        required_args={"url"},
        validators=[_v_url("url")],
    ))
    registry.register_tool("update_world_model", ToolSpec(
        fn=update_world_model_tool,
        task_type="GenericTask",
        roles_allowed={"Planner", "Observer", "Worker"},
        timeout_s=5, idempotent=True,
        required_args={"key", "value"},
    ))
    registry.register_tool("query_long_term_memory", ToolSpec(
        fn=query_long_term_memory_tool,
        task_type="GenericTask",
        roles_allowed={"Planner", "Observer"},
        timeout_s=8, idempotent=True,
        required_args={"query_text"},
    ))

    # Reporting
    registry.register_tool("create_pdf", ToolSpec(
        fn=create_pdf_tool,
        task_type="Reporting",
        roles_allowed={"Worker", "Planner"},
        timeout_s=20, idempotent=True,
        required_args={"filename", "text_content"},
    ))
    registry.register_tool("generate_report_pdf", ToolSpec(
        fn=generate_report_pdf_tool,
        task_type="Reporting",
        roles_allowed={"Worker", "Planner"},
        timeout_s=25, idempotent=True,
        required_args={"title", "sections"},
    ))

    # Planner assist
    registry.register_tool("shuffle_roles_and_tasks", ToolSpec(
        fn=shuffle_roles_and_tasks_tool,
        task_type="PlannerAssist",
        roles_allowed={"Planner"},
        timeout_s=8, idempotent=True,
        required_args={"stagnant_agents"},
    ))


# ------------------------------------------------------------------------------
# Convenience aliases + optional proxy invocation
# ------------------------------------------------------------------------------
# simple aliases some code may import
kubernetes_pod_metrics = kubernetes_pod_metrics_tool
k8s_scale = k8s_scale_tool

def invoke_tool(name: str, args: Dict[str, Any]) -> dict:
    """
    Optional convenience proxy: if a global tool_registry.tool_registry instance
    exists, use it to invoke. Otherwise return an error.
    Imported lazily to avoid circular imports.
    """
    try:
        from tool_registry import tool_registry  # type: ignore
    except Exception:
        return {"ok": False, "error": "Central tool registry not available; call register_tools_into(...) and use registry.invoke()."}
    try:
        return tool_registry.invoke(name, args or {})
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------------------
# Public exports
# ------------------------------------------------------------------------------
__all__ = [
    # registration
    "register_tools_into",
    "invoke_tool",
    # tools
    "get_system_cpu_load_tool",
    "get_system_resource_usage_tool",
    "disk_usage_tool",
    "top_processes_tool",
    "measure_responsiveness_tool",
    "kubernetes_pod_metrics_tool",
    "k8s_scale_tool",
    "initiate_network_scan_tool",
    "deploy_recovery_protocol_tool",
    "analyze_threat_signature_tool",
    "isolate_network_segment_tool",
    "extract_iocs_tool",
    "hash_text_tool",
    "get_environmental_data_tool",
    "web_search_tool",
    "read_webpage_tool",
    "update_world_model_tool",
    "query_long_term_memory_tool",
    "analyze_text_sentiment_tool",
    "redact_pii_tool",
    "create_pdf_tool",
    "generate_report_pdf_tool",
    "shuffle_roles_and_tasks_tool",
    "prometheus_query_tool",
    "prometheus_range_query_tool",
    # helpers/validators you may want elsewhere
    "_parse_kubectl_top_pods",
    "_v_has_namespace",
    "_v_k8s_scale_args",
    "_v_url",
    "_v_enum",
    "_v_ipv4",
    # aliases
    "kubernetes_pod_metrics",
    "k8s_scale",
]


def register_tools_into(registry):
    """compat shim; ToolRegistry now self-registers tools"""
    return
