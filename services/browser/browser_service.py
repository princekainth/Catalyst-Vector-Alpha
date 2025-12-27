"""Browser Service - Full web access for CVA agents."""
import json
import requests
from typing import Optional, Dict
from urllib.parse import quote_plus
from .cache import URLCache
from .extractor import ContentExtractor
from .rate_limiter import RateLimiter

class BrowserService:
    def __init__(self, cache_ttl: int = 3600, timeout: int = 30, user_agent: str = None):
        self.cache = URLCache(ttl_seconds=cache_ttl)
        self.extractor = ContentExtractor()
        self.rate_limiter = RateLimiter()
        self.timeout = timeout
        self.user_agent = user_agent or "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0"
        print(f"[BrowserService] Initialized (cache_ttl={cache_ttl}s)")
    
    def web_fetch(self, url: str, use_cache: bool = True, extract: bool = True) -> dict:
        """Fetch URL and return content."""
        if use_cache:
            cached = self.cache.get(url)
            if cached:
                if extract:
                    ext = self.extractor.extract(cached["content"], url)
                    cached["extracted"] = {"title": ext.title, "text": ext.text, "links": ext.links, "code_blocks": ext.code_blocks}
                return cached
        
        self.rate_limiter.wait_if_needed(url)
        self.rate_limiter.record_request(url)
        
        try:
            resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=self.timeout)
            result = {"status": "ok", "url": resp.url, "status_code": resp.status_code, "content": resp.text}
            self.cache.set(url, resp.text, resp.headers.get("Content-Type", ""), resp.status_code)
            if extract:
                ext = self.extractor.extract(resp.text, url)
                result["extracted"] = {"title": ext.title, "text": ext.text, "links": ext.links, "code_blocks": ext.code_blocks}
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def web_search(self, query: str, num_results: int = 5) -> dict:
        """Search web using DuckDuckGo."""
        import re
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            resp = requests.get(search_url, headers={"User-Agent": self.user_agent}, timeout=self.timeout)
            results = []
            for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', resp.text):
                url, title = m.groups()
                if "uddg=" in url:
                    from urllib.parse import unquote
                    match = re.search(r'uddg=([^&]+)', url)
                    if match: url = unquote(match.group(1))
                results.append({"title": title.strip(), "url": url})
                if len(results) >= num_results: break
            return {"status": "ok", "query": query, "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def web_api(self, url: str, method: str = "GET", data: dict = None, headers: dict = None) -> dict:
        """Make REST API call."""
        try:
            req_headers = {"User-Agent": self.user_agent}
            if headers: req_headers.update(headers)
            self.rate_limiter.wait_if_needed(url)
            self.rate_limiter.record_request(url)
            resp = requests.request(method=method.upper(), url=url, json=data if method.upper() in ("POST","PUT") else None,
                params=data if method.upper() == "GET" and data else None, headers=req_headers, timeout=self.timeout)
            try: resp_data = resp.json()
            except: resp_data = resp.text
            return {"status": "ok", "url": url, "status_code": resp.status_code, "response": resp_data}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def learn_and_store(self, url: str, topic: str, memory) -> dict:
        """Fetch URL, extract knowledge, store in memory for Teachers to share with Students."""
        result = self.web_fetch(url)
        if result["status"] != "ok": return result
        summary = self.extractor.summarize_for_memory(result["content"], url, topic)
        if hasattr(memory, "add_memory"):
            memory.add_memory(memory_type="WebKnowledge", content=summary, source_agent="BrowserService")
        return {"status": "ok", "url": url, "topic": topic, "title": summary["title"], "stored": True}
