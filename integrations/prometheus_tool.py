from __future__ import annotations
import os, logging, requests
from typing import Dict, Any, Optional

class PrometheusMetrics:
    """
    Minimal Prometheus HTTP API client.
    Requires PROMETHEUS_URL, e.g. http://localhost:9090
    """
    def __init__(self, base_url: Optional[str] = None, logger: Optional[logging.Logger] = None, timeout=5):
        self.base_url = (base_url or os.getenv("PROMETHEUS_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("PROMETHEUS_URL not set")
        self.timeout = timeout
        self.log = logger or logging.getLogger("PrometheusMetrics")

    def _query(self, promql: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/query"
        r = requests.get(url, params={"query": promql}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {data}")
        return data

    def cpu_percent_avg_5m(self, instance: Optional[str] = None) -> float:
        # 100 - idle, using node_exporter. Adjust if your metrics differ.
        filt = f'{{instance="{instance}"}}' if instance else ""
        promql = f"100 - (avg by (instance) (irate(node_cpu_seconds_total{filt}[5m])) * 100)"
        data = self._query(promql)
        results = data.get("data", {}).get("result", [])
        if not results:
            return 0.0
        vals = [float(r["value"][1]) for r in results]
        return sum(vals) / max(1, len(vals))

    def http_p95_ms_5m(self, service_label: str) -> float:
        promql = (
            f'histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{{service="{service_label}"}}[5m]))) * 1000'
        )
        data = self._query(promql)
        results = data.get("data", {}).get("result", [])
        return float(results[0]["value"][1]) if results else 0.0
