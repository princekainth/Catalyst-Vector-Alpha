"""Rate Limiter - Per-domain rate limiting to avoid bans."""
import time
from collections import defaultdict
from urllib.parse import urlparse
from threading import Lock

class RateLimiter:
    DEFAULT_LIMITS = {
        "google.com": 10, "stackoverflow.com": 20, "github.com": 30,
        "hub.docker.com": 20, "kubernetes.io": 30, "default": 30
    }
    
    def __init__(self, custom_limits: dict = None):
        self.limits = {**self.DEFAULT_LIMITS, **(custom_limits or {})}
        self.requests = defaultdict(list)
        self.lock = Lock()
    
    def _get_domain(self, url: str) -> str:
        try:
            domain = urlparse(url).netloc.lower()
            return domain[4:] if domain.startswith("www.") else domain
        except:
            return "unknown"
    
    def _get_limit(self, domain: str) -> int:
        if domain in self.limits:
            return self.limits[domain]
        for known, limit in self.limits.items():
            if domain.endswith("." + known):
                return limit
        return self.limits.get("default", 30)
    
    def wait_if_needed(self, url: str) -> float:
        domain = self._get_domain(url)
        limit = self._get_limit(domain)
        with self.lock:
            now = time.time()
            self.requests[domain] = [t for t in self.requests[domain] if now - t < 60]
            if len(self.requests[domain]) < limit:
                return 0.0
            oldest = min(self.requests[domain])
            wait_time = 60 - (now - oldest) + 0.1
        if wait_time > 0:
            time.sleep(wait_time)
            return wait_time
        return 0.0
    
    def record_request(self, url: str):
        domain = self._get_domain(url)
        with self.lock:
            self.requests[domain].append(time.time())
