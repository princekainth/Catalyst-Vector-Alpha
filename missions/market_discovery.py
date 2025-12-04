import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
from urllib.parse import urlparse

import yaml
from llm_ollama import OllamaLLMIntegration

from tool_registry import ToolRegistry, tool_registry as GLOBAL_TOOL_REGISTRY

log = logging.getLogger("MarketResearcher")


@dataclass
class Competitor:
    id: str
    name: str
    url: str
    categories: List[str]
    source_queries: List[str]
    snippet_content: Optional[str] = None
    snippet_analysis: Optional[Dict[str, Any]] = None
    suspected_competitor: bool = False
    services_page: Optional[str] = None
    services_excerpt: Optional[str] = None
    services_content: Optional[str] = None
    barrie_confirmed: bool = False
    extracted_prices: Optional[Dict[str, Any]] = None
    llm_raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "categories": sorted(set(self.categories)),
            "source_queries": sorted(set(self.source_queries)),
            "snippet_content": self.snippet_content,
            "snippet_analysis": self.snippet_analysis,
            "suspected_competitor": self.suspected_competitor,
            "services_page": self.services_page,
            "services_excerpt": self.services_excerpt,
            "services_content": self.services_content,
            "barrie_confirmed": self.barrie_confirmed,
            "extracted_prices": self.extracted_prices,
            "llm_raw_response": self.llm_raw_response,
        }


class MarketResearcher:
    """Discover competitors via web_search + read_webpage and persist the findings."""

    def __init__(
        self,
        config_path: str = "config/market_research.yaml",
        tool_registry: Optional[ToolRegistry] = None,
        persistence_path: str = "persistence_data/competitors_db.json",
    ):
        self.config_path = Path(config_path)
        self.persistence_path = Path(persistence_path)
        self.tool_registry = tool_registry or GLOBAL_TOOL_REGISTRY
        self.llm = OllamaLLMIntegration()
        self.config = self._load_config()
        self.location: str = self.config.get("location", "").strip()
        self.my_business: str = self.config.get("my_business", "").strip()
        self.search_categories: List[str] = self.config.get("search_categories", [])
        self.data_to_extract: List[str] = self.config.get("data_to_extract", [])

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing market research config: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _ensure_persistence_dir(self):
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_valid_url(url: Optional[str]) -> bool:
        if not url:
            return False
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return (name or "").strip()

    @staticmethod
    def _dedupe_key(name: str, url: str) -> str:
        domain = urlparse(url).netloc.lower()
        return f"{domain}|{name.strip().lower()}"

    @staticmethod
    def _snippet_suspect(snippet: str, category: str) -> bool:
        text = (snippet or "").lower()
        geo_hits = any(token in text for token in ["barrie", "l4n", "l4m", "simcoe"])
        cat_hits = any(token in text for token in [category.lower(), "spa", "laser", "microneedling", "facial", "pmu"])
        return geo_hits and cat_hits

    @staticmethod
    def _is_aggregator(url: str) -> bool:
        domain = urlparse(url).netloc.lower()
        bad_fragments = [
            "yelp.", "tripadvisor.", "threebestrated", "fresha.", "craigslist.",
            "royallepage.", "kijiji.", "cylex-", "cylex.", "iglobal.",
            "storestip.", "vk.com", "tiktok.com", "facebook.", "instagram.",
            "google.", "maps.google", "bing.com", "booking.com"
        ]
        return any(b in domain for b in bad_fragments)

    @staticmethod
    def _extract_services_snippet(content: str, window: int = 180) -> Optional[str]:
        text = content or ""
        lower = text.lower()
        idx = lower.find("service")
        if idx == -1:
            return None
        start = max(0, idx - window // 2)
        end = min(len(text), idx + window // 2)
        snippet = text[start:end].strip()
        return " ".join(snippet.split())

    @staticmethod
    def _clean_html_text(html: str) -> str:
        text = re.sub("<[^<]+?>", " ", html or "")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _extract_prices(text: str) -> List[str]:
        """Heuristic price extraction from text."""
        if not text:
            return []
        prices: List[str] = []
        service_keywords = r"(laser|wax|brazilian|facial|hydrafacial|pmu|brow|thread|skin|spa|hair removal)"
        price_pat = re.compile(rf"{service_keywords}[^\n]{{0,80}}?(?:\$|cad)\s*\d{{2,4}}(?:\.\d{{2}})?", re.IGNORECASE)
        for m in price_pat.finditer(text):
            hit = " ".join(m.group(0).split())
            prices.append(hit)
        # Also catch standalone prices with currency
        standalone_pat = re.compile(r"(?:\$|cad)\s*\d{2,4}(?:\.\d{2})?", re.IGNORECASE)
        for m in standalone_pat.finditer(text):
            hit = " ".join(m.group(0).split())
            prices.append(hit)
        # Dedupe preserving order
        seen = set()
        deduped = []
        for p in prices:
            if p.lower() in seen:
                continue
            seen.add(p.lower())
            deduped.append(p)
        return deduped

    @staticmethod
    def _extract_links(html: str, base_url: str) -> List[str]:
        """Pull hrefs and normalize to absolute URLs."""
        if not html:
            return []
        hrefs = re.findall(r'href=[\"\\\'](.*?)[\"\\\']', html, flags=re.IGNORECASE)
        urls = []
        base = base_url
        for h in hrefs:
            if h.startswith("#") or h.lower().startswith("mailto:") or h.lower().startswith("tel:"):
                continue
            try:
                absolute = urljoin(base, h)
            except Exception:
                absolute = h
            urls.append(absolute)
        # dedupe preserving order
        deduped = []
        seen = set()
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            deduped.append(u)
        return deduped

    @staticmethod
    def _extract_links_with_text(html: str, base_url: str, keywords: List[str]) -> List[str]:
        """Extract links whose anchor text contains keyword hits."""
        if not html or not keywords:
            return []
        anchors = re.findall(r'<a[^>]+href=[\"\\\'](.*?)[\"\\\'][^>]*>(.*?)</a>', html, flags=re.IGNORECASE | re.DOTALL)
        urls = []
        for href, text in anchors:
            txt = re.sub("<[^<]+?>", " ", text or "")
            txt_l = txt.lower()
            if not any(k in txt_l for k in keywords):
                continue
            try:
                absolute = urljoin(base_url, href)
            except Exception:
                absolute = href
            urls.append(absolute)
        deduped = []
        seen = set()
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            deduped.append(u)
        return deduped

    def _headless_fetch(self, url: str, timeout_ms: int = 20000) -> Optional[str]:
        """Render the page with Playwright if available to bypass basic blocking."""
        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:
            log.debug("Headless fetch skipped (playwright not available): %s", e)
            return None

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    viewport={"width": 1280, "height": 720},
                )
                page = context.new_page()
                page.goto(url, timeout=timeout_ms, wait_until="networkidle")
                html = page.content()
                browser.close()
                return self._clean_html_text(html)
        except Exception as e:
            log.warning("Headless fetch failed for %s: %s", url, e)
            return None

    def _call_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        res = self.tool_registry.safe_call(name, **kwargs)
        if not isinstance(res, dict):
            return {"status": "error", "error": f"Invalid response from {name}: {res}"}
        return res

    def discover_competitors(self, max_results_per_category: int = 5) -> Dict[str, Any]:
        """Iterate categories, run web searches, dedupe, and persist a competitor DB."""
        if not self.search_categories or not self.location:
            raise ValueError("Config must include 'search_categories' and 'location'.")

        unique: OrderedDict[str, Competitor] = OrderedDict()
        category_index: Dict[str, List[str]] = {}

        for category in self.search_categories:
            query = f"{category} {self.location}"
            resp = self._call_tool("web_search", query=query, max_results=max_results_per_category)
            tool_output = resp.get("data", {})
            if isinstance(tool_output, dict) and "data" in tool_output and "results" in tool_output["data"]:
                results = tool_output["data"]["results"]
            else:
                results = tool_output.get("results", [])
            category_hits: List[str] = []

            for hit in results:
                name = self._normalize_name(hit.get("title") or hit.get("body") or "")
                url = hit.get("href") or hit.get("url")
                snippet = hit.get("body") or ""

                if self._is_aggregator(url or ""):
                    continue

                if not name and url:
                    name = urlparse(url).netloc
                if not (name and url and self._is_valid_url(url)):
                    continue

                snippet_analysis = self.analyze_competitor(snippet, url=url) if snippet else {"is_barrie_competitor": False, "prices": []}
                suspected = self._snippet_suspect(snippet, category)

                key = self._dedupe_key(name, url)
                if key not in unique:
                    unique[key] = Competitor(
                        id=key,
                        name=name,
                        url=url,
                        categories=[category],
                        source_queries=[query],
                        snippet_content=snippet,
                        snippet_analysis=snippet_analysis,
                        suspected_competitor=suspected,
                    )
                else:
                    unique[key].categories.append(category)
                    unique[key].source_queries.append(query)
                    if snippet and not unique[key].snippet_content:
                        unique[key].snippet_content = snippet
                        unique[key].snippet_analysis = snippet_analysis
                    unique[key].suspected_competitor = unique[key].suspected_competitor or suspected

                if key not in category_hits:
                    category_hits.append(key)

            category_index[category] = category_hits

        ordered_keys = list(unique.keys())
        self._enrich_services_pages(unique, ordered_keys, limit=min(len(ordered_keys), 5))
        filtered_pool, category_index = self._apply_llm_filter(unique, ordered_keys, category_index)

        competitors_payload = [c.to_dict() for c in filtered_pool.values()]
        payload = {
            "location": self.location,
            "my_business": self.my_business,
            "search_categories": self.search_categories,
            "data_to_extract": self.data_to_extract,
            "total_competitors": len(competitors_payload),
            "category_index": category_index,
            "competitors": competitors_payload,
        }

        self._ensure_persistence_dir()
        with self.persistence_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return payload

    def _enrich_services_pages(self, pool: OrderedDict[str, Competitor], order: List[str], limit: int = 10):
        """Attempt to read a services page for a batch of competitors."""
        target_keys = order[:limit] if limit else order
        for key in target_keys:
            comp = pool.get(key)
            if not comp or not self._is_valid_url(comp.url):
                continue

            base_data = self._get_page_data(comp.url)
            base_content = base_data.get("text", "")
            links = base_data.get("links", [])
            pricing_links = base_data.get("pricing_links", [])
            # Filter links to same domain and relevant keywords
            filtered_links = []
            for l in links:
                if self._is_aggregator(l):
                    continue
                if urlparse(l).netloc and urlparse(l).netloc != urlparse(comp.url).netloc:
                    continue
                lower_l = l.lower()
                if any(k in lower_l for k in ["service", "price", "treatment", "laser", "wax", "facial", "book", "appointment", "menu", "rate"]):
                    filtered_links.append(l)

            candidates = pricing_links + self._candidate_service_urls(comp.url) + filtered_links

            for candidate in candidates:
                content = self._get_page_data(candidate).get("text", "")
                if not content:
                    continue
                snippet = self._extract_services_snippet(content)
                comp.services_page = candidate
                comp.services_excerpt = snippet
                comp.services_content = content
                if not comp.extracted_prices:
                    comp.extracted_prices = self._extract_prices(content)
                break

    def _get_page_data(self, url: str) -> Dict[str, Any]:
        """Fetch raw HTML + text + links using requests or headless fallback."""
        html = ""
        text = ""
        links: List[str] = []
        pricing_links: List[str] = []

        # Try direct requests to preserve HTML for link extraction
        try:
            import requests

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.google.com/",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code == 200 and r.text:
                html = r.text
                text = self._clean_html_text(html)
        except Exception:
            pass

        # Fallback to headless if needed
        if (not html or len(text) < 300) and not self._is_aggregator(url):
            rendered_html = self._headless_fetch(url)
            if rendered_html:
                text = self._clean_html_text(rendered_html)
                html = rendered_html

        # Fallback to tool if still empty
        if not text:
            resp = self._call_tool("read_webpage", url=url)
            if resp.get("status") == "ok":
                text = (resp.get("data") or {}).get("content", "") or ""

        # Extract links from HTML if we have it
        if html:
            links = self._extract_links(html, url)
            pricing_links = self._extract_links_with_text(
                html,
                url,
                ["price", "pricing", "service", "services", "book", "booking", "appointment", "menu", "rate", "laser", "wax", "facial"],
            )

        return {"text": text or "", "links": links, "pricing_links": pricing_links}

    @staticmethod
    def _candidate_service_urls(base_url: str) -> List[str]:
        """Generate likely service/pricing URLs for a business site."""
        base = base_url.rstrip("/")
        paths = [
            "",
            "/services",
            "/service",
            "/pricing",
            "/prices",
            "/treatments",
            "/treatment",
            "/menu",
            "/rates",
            "/fees",
            "/services-menu",
            "/service-menu",
            "/laser-hair-removal",
            "/laser",
            "/spa-services",
            "/medi-spa",
            "/medi-spa-services",
            "/skin-care",
            "/skin",
            "/hair-removal",
            "/laser-hair-removal-pricing",
            "/facials",
            "/waxing",
            "/brazilian-wax",
            "/waxing-services",
            "/waxing-prices",
            "/book",
            "/book-online",
            "/book-now",
            "/booking",
            "/appointments",
        ]
        urls = []
        for p in paths:
            if p:
                urls.append(f"{base}{p}")
                urls.append(f"{base}{p}.html")
            else:
                urls.append(base_url)
        return list(dict.fromkeys(urls))  # dedupe, preserve order

    def _apply_llm_filter(
        self,
        pool: OrderedDict[str, Competitor],
        order: List[str],
        category_index: Dict[str, List[str]],
    ) -> tuple[OrderedDict[str, Competitor], Dict[str, List[str]]]:
        """Analyze competitors and retain only those confirmed or strongly suspected in Barrie."""
        confirmed_keys: List[str] = []
        for key in order:
            comp = pool.get(key)
            if not comp:
                continue
            if self._is_aggregator(comp.url):
                continue
            content = comp.services_content or comp.services_excerpt
            snippet_is_comp = False
            snippet_prices = None
            if comp.snippet_analysis:
                snippet_is_comp = bool(comp.snippet_analysis.get("is_barrie_competitor")) or comp.suspected_competitor
                snippet_prices = comp.snippet_analysis.get("prices")
            else:
                snippet_is_comp = comp.suspected_competitor

            is_competitor = False
            analysis = None

            if content:
                analysis = self.analyze_competitor(content, url=comp.url)
                comp.llm_raw_response = analysis.get("raw")
                comp.extracted_prices = analysis.get("prices")
                comp.barrie_confirmed = bool(analysis.get("is_barrie_competitor"))
                is_competitor = comp.barrie_confirmed

                if len(content) < 500 and snippet_is_comp:
                    # fallback to snippet truth if page is thin
                    is_competitor = True
                    comp.barrie_confirmed = True
                    if not comp.extracted_prices and snippet_prices:
                        comp.extracted_prices = snippet_prices
                # Fallback to regex price extraction if LLM returned none
                if not comp.extracted_prices:
                    regex_prices = self._extract_prices(content)
                    if regex_prices:
                        comp.extracted_prices = regex_prices
            else:
                is_competitor = snippet_is_comp
                comp.barrie_confirmed = comp.barrie_confirmed or snippet_is_comp
                if snippet_prices and not comp.extracted_prices:
                    comp.extracted_prices = snippet_prices

            print(f"Checked {comp.name}: Is Competitor? {is_competitor}")

            if is_competitor:
                confirmed_keys.append(key)

        filtered_pool = OrderedDict((k, pool[k]) for k in confirmed_keys)
        filtered_index: Dict[str, List[str]] = {
            cat: [k for k in keys if k in filtered_pool] for cat, keys in category_index.items()
        }
        return filtered_pool, filtered_index

    def _load_saved(self) -> Optional[Dict[str, Any]]:
        if not self.persistence_path.exists():
            return None
        try:
            with self.persistence_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Failed to read existing competitors DB: %s", e)
            return None

    def analyze_competitor(self, content: str, url: str = "") -> Dict[str, Any]:
        """Use LLM to confirm Barrie location and extract prices from services content."""
        if not content:
            return {"raw": "", "parsed": None, "is_barrie_competitor": False, "prices": []}

        print(f"[DEBUG] Analyzing {url or 'unknown URL'} (Content len: {len(content)})")
        print(f"[DEBUG] Content length for {url or 'unknown URL'}: {len(content)}")

        prompt = (
            "Analyze this website text. "
            "1. Is this a beauty/spa business physically located in Barrie, Ontario? (Return TRUE/FALSE). "
            "If the location is not explicitly stated but the content mentions '705' area code or 'Simcoe', assume it is Barrie. "
            "2. Extract any prices for Laser, Microneedling, or Facials. "
            "Return ONLY valid JSON. No markdown formatting. "
            "Format strictly as: { 'is_barrie_competitor': bool, 'prices': [...] }\n\n"
            f"Text:\n{content[:4000]}"
        )

        raw = ""
        try:
            raw = self.llm.generate_text(prompt=prompt, max_tokens=350, temperature=0.2, json_mode=True)
        except Exception as e:
            log.warning("LLM analysis failed: %s", e)
            return {"raw": raw, "parsed": None, "is_barrie_competitor": False, "prices": []}

        print(f"[DEBUG] LLM Raw Response: {raw}")

        parsed = None
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                parsed = json.loads(json_str)
            except Exception:
                parsed = None
        if parsed is None:
            parsed = self._parse_json_response(raw)
        is_barrie = False
        prices: Any = []
        if parsed:
            is_barrie_val = parsed.get("is_barrie_competitor")
            if isinstance(is_barrie_val, str):
                is_barrie = is_barrie_val.strip().lower() in {"true", "yes", "y"}
            else:
                is_barrie = bool(is_barrie_val)
            prices_field = parsed.get("prices")
            if isinstance(prices_field, list):
                prices = prices_field
            elif isinstance(prices_field, dict):
                prices = prices_field

        return {
            "raw": raw,
            "parsed": parsed,
            "is_barrie_competitor": is_barrie,
            "prices": prices,
        }

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON extraction from LLM output."""
        if not raw:
            return None
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                return json.loads(cleaned)
            except Exception:
                pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                return None
        return None

    def format_report(self, data: Optional[Dict[str, Any]] = None, top_n_per_category: int = 3) -> str:
        """Format a text summary of discovered competitors."""
        payload = data or self._load_saved()
        if not payload:
            return "No competitor data available. Run discover_competitors() first."

        comps = [c for c in payload.get("competitors", []) if isinstance(c, dict)]
        confirmed = [c for c in comps if c.get("barrie_confirmed")]
        comp_lookup = {c.get("id"): c for c in confirmed if isinstance(c, dict)}
        category_index = payload.get("category_index", {})

        lines = [
            f"Market research for {payload.get('location', self.location)}",
            f"My business: {payload.get('my_business', self.my_business)}",
            f"Confirmed Barrie competitors: {len(confirmed)}",
            "",
        ]

        for category in payload.get("search_categories", self.search_categories):
            ids = category_index.get(category, [])
            top_competitors = [comp_lookup[i] for i in ids if i in comp_lookup][:top_n_per_category]
            if top_competitors:
                formatted = []
                for comp in top_competitors:
                    prices = comp.get("extracted_prices")
                    price_str = "prices not found"
                    if isinstance(prices, dict):
                        bits = [f"{k}: {v}" for k, v in prices.items()]
                        if bits:
                            price_str = "; ".join(bits)
                    elif isinstance(prices, list):
                        bits = [str(p) for p in prices if p]
                        if bits:
                            price_str = "; ".join(bits)
                    formatted.append(f"{comp.get('name')} (Prices: {price_str})")
                lines.append(f"{category}: {', '.join(formatted)}")
            else:
                lines.append(f"{category}: no competitors captured.")

        services_hits = [c for c in confirmed if c.get("services_page")]
        if services_hits:
            lines.append("")
            lines.append("Services pages sampled:")
            for comp in services_hits[:3]:
                name = comp.get("name")
                url = comp.get("services_page")
                excerpt = comp.get("services_excerpt") or ""
                lines.append(f"- {name} → {url}")
                if excerpt:
                    lines.append(f"  Snippet: {excerpt}")

        if payload.get("data_to_extract"):
            lines.append("")
            lines.append(f"Data targets: {', '.join(payload['data_to_extract'])}")

        return "\n".join(lines)


if __name__ == "__main__":
    researcher = MarketResearcher()
    data = researcher.discover_competitors()
    print(researcher.format_report(data))
