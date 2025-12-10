import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
from urllib.parse import urlparse, urljoin

import yaml
from llm_ollama import OllamaLLMIntegration
from tools_headless import fetch_page_content

from tool_registry import ToolRegistry, tool_registry as GLOBAL_TOOL_REGISTRY

log = logging.getLogger("MarketResearcher")

AGGREGATOR_DOMAINS = {
    "yellowpages.ca",
    "bestprosintown.com",
    "ruli.com",
    "groupon.com",
    "bing.com",
    "duckduckgo.com",
}

DOMAIN_BLACKLIST = {
    "arcticspasbarrie.ca",
    "nano-reef.com",
    "bestbuyersguide.org",
    "canadianorglist.com",
    "mapquest.com",
    "welltipsforyou.com",
    "findprivateclinics.ca",
    "groupon.com",
}

ALWAYS_COMPETITOR_DOMAINS = {
    "glowdayspa.ca",
    "spalumina.com",
    "revivemedspa.ca",
    "naturalbalance-dayspa.com",
    "mottalashhouse.com",
    "waxingwithanna.com",
    "browhousebarrie.com",
    "joinusgorgeous.com",
    "lashnnailroom.ca",
    "forever-flawless.ca",
    "bonitalaser.ca",
    "thelaserbar.ca",
}

BOOKING_DOMAINS = [
    "fresha.com",
    "janeapp.com",
    "mindbodyonline.com",
    "vagaro.com",
    "booker.com",
    "schedulicity.com",
]


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
    def _collect_numeric_prices(prices: Any) -> List[float]:
        """Extract numeric price values from price structures (currency-sign only)."""
        nums: List[float] = []
        pat = re.compile(r"(?:\$|cad)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
        if isinstance(prices, dict):
            for v in prices.values():
                items = v if isinstance(v, list) else [v]
                for item in items:
                    if not item:
                        continue
                    for m in pat.findall(str(item)):
                        try:
                            nums.append(float(m))
                        except Exception:
                            continue
        elif isinstance(prices, list):
            for item in prices:
                if not item:
                    continue
                for m in pat.findall(str(item)):
                    try:
                        nums.append(float(m))
                    except Exception:
                        continue
        return nums

    @staticmethod
    def _snippet_suspect(snippet: str, category: str) -> bool:
        text = (snippet or "").lower()
        geo_hits = any(token in text for token in ["barrie", "l4n", "l4m", "simcoe"])
        cat_hits = any(token in text for token in [category.lower(), "spa", "laser", "microneedling", "facial", "pmu"])
        return geo_hits and cat_hits

    @staticmethod
    def _is_aggregator(url: str) -> bool:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        if domain in AGGREGATOR_DOMAINS:
            return True
        bad_fragments = [
            "yelp.",
            "tripadvisor.",
            "threebestrated",
            "fresha.",
            "craigslist.",
            "royallepage.",
            "kijiji.",
            "cylex-",
            "cylex.",
            "iglobal.",
            "storestip.",
            "vk.com",
            "tiktok.com",
            "facebook.",
            "instagram.",
            "google.",
            "maps.google",
            "booking.com",
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

    @staticmethod
    def _normalized_domain(url: str) -> str:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host

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
        """Attempt to read a services page for a batch of competitors using spider strategy."""
        target_keys = order[:limit] if limit else order
        for key in target_keys:
            comp = pool.get(key)
            if not comp or not self._is_valid_url(comp.url):
                continue

            # Step 1: fetch home page (headless) and collect links
            base_data = self._get_page_data(comp.url)
            base_text = base_data.get("text", "")
            base_links = base_data.get("links", []) or []

            # Step 2: choose best link by priority
            keywords = ["price", "pricing", "rate", "cost", "service", "services", "menu", "treatment", "book"]
            negative_terms = ["ski", "golf", "resort", "hotel", "room", "lift", "ticket"]
            boost_terms = ["aesthetics", "laser", "facial", "skin", "body", "treatment", "spa", "wax"]
            best_link = None

            # Priority 1: booking engines (any domain)
            for link in base_links:
                href = (link.get("href") or "").strip()
                if not href:
                    continue
                lower_href = href.lower()
                if any(engine in lower_href for engine in BOOKING_DOMAINS):
                    best_link = href
                    log.info(f"[INFO] Found external booking engine: {href} - Following immediately.")
                    break

            # Priority 2: keyword matches on same domain if no booking engine found
            if not best_link:
                scored = []
                for link in base_links:
                    href = (link.get("href") or "").strip()
                    text = (link.get("text") or "").lower()
                    target = href.lower() + " " + text
                    if not href or self._is_aggregator(href):
                        continue
                    if urlparse(href).netloc and urlparse(href).netloc != urlparse(comp.url).netloc:
                        # allow offsite for booking only; otherwise skip
                        continue
                    if any(neg in target for neg in negative_terms) and "spa" not in target:
                        continue
                    score = 0
                    if any(k in target for k in keywords):
                        score += 2
                    if any(b in target for b in boost_terms):
                        score += 1
                    if score > 0:
                        scored.append((score, href))
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    best_link = scored[0][1]

            combined_text = base_text
            if best_link:
                best_data = self._get_page_data(best_link)
                best_text = best_data.get("text", "")
                if best_text:
                    combined_text = base_text + "\n" + best_text
                    comp.services_page = best_link
                    log.info(f"[SUCCESS] Fetched booking engine page: {best_link}, got {len(best_text)} chars")
            else:
                combined_text = base_text
                comp.services_page = comp.url

            if not combined_text:
                continue

            comp.services_excerpt = self._extract_services_snippet(combined_text)
            comp.services_content = combined_text
            if not comp.extracted_prices:
                comp.extracted_prices = self._extract_prices(combined_text)

    def _get_page_data(self, url: str) -> Dict[str, Any]:
        """Fetch raw HTML + text + links using headless fetch."""
        try:
            res = fetch_page_content(url)
        except Exception as e:
            log.warning("Headless fetch_page_content failed for %s: %s", url, e)
            res = {}

        html = res.get("content", "") if isinstance(res, dict) else ""
        links = res.get("links", []) if isinstance(res, dict) else []
        text = self._clean_html_text(html)
        if not text:
            resp = self._call_tool("read_webpage", url=url)
            if resp.get("status") == "ok":
                text = (resp.get("data") or {}).get("content", "") or ""
        return {"text": text or "", "links": links}

    @staticmethod
    def _looks_like_barrie_service(text: str, url: str) -> bool:
        blob = f"{text} {url}".lower()
        if all(token not in blob for token in ["barrie", "l4n", "l4m", "simcoe"]):
            return False
        service_keywords = [
            "laser",
            "spa",
            "medspa",
            "wax",
            "brazilian",
            "thread",
            "microneedling",
            "facial",
            "hydrafacial",
            "brow",
            "pmu",
            "permanent makeup",
            "cosmetic tattoo",
            "lash",
        ]
        return any(k in blob for k in service_keywords)

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
        """Use LLM to confirm Barrie location and extract structured services + prices."""
        # Hard filter: never treat obviously irrelevant domains as competitors.
        hostname = ""
        if url:
            try:
                hostname = urlparse(url).hostname or ""
            except Exception:
                hostname = ""
        if any(hostname.endswith(d) for d in DOMAIN_BLACKLIST):
            return {
                "raw": "",
                "parsed": None,
                "is_barrie_competitor": False,
                "prices": {},
            }

        if not content:
            return {
                "raw": "",
                "parsed": None,
                "is_barrie_competitor": False,
                "prices": {},
            }

        print(f"[DEBUG] Analyzing {url or 'unknown URL'} (Content len: {len(content)})")
        print(f"[DEBUG] Content length for {url or 'unknown URL'}: {len(content)}")

        # Truncate huge content to a middle slice; booking engines often sit there.
        max_chars = 20000
        if len(content) > max_chars * 2:
            start = (len(content) // 2) - (max_chars // 2)
            end = start + max_chars
            content = content[start:end]

        prompt = (
            "Analyze this website text.\n"
            "You are classifying local beauty/medical spa competitors in Barrie, Ontario.\n"
            "Prefer <body> content and visible text (about/contact/services) over navigation/SEO blobs when detecting services and location.\n"
            "Step 1: Decide if this is a beauty/skin/laser/spa/PMU business serving Barrie, Ontario. "
            "Return TRUE/FALSE for is_barrie_competitor based only on business type and Barrie location cues. "
            "This must NOT depend on prices. Do NOT count hot tub retailers, equipment dealers, or generic directories/articles as competitors.\n"
            "Step 2: Independently extract a structured list of services and prices ONLY for client-facing beauty/skin/spa services "
            "(Laser hair removal, waxing/Brazilian, facials/Hydrafacial, microneedling, PMU/brows, threading, skin treatments). For each service, return:\n"
            "  - raw_name: exact service name as shown (e.g. 'Brazilian Wax', 'Full Legs', 'Underarms Laser').\n"
            "  - normalized_name: short snake_case identifier (e.g. 'brazilian', 'full_legs', 'underarms_laser').\n"
            "  - category: one of ['laser', 'waxing', 'brazilian', 'facial', 'hydrafacial', 'pmu', 'brows', 'threading', 'other'].\n"
            "  - price: numeric price (no currency symbol), or null if no price is visible.\n"
            "  - currency: 'CAD' if you can infer it, otherwise null.\n"
            "Ignore prices for gift cards/gift certificates, deposits/fees, products/retail, parking, hotel rooms, lift tickets, food/drinks. "
            "If no valid service prices are visible, return an empty services list (or price=null) but do NOT change is_barrie_competitor because of missing prices.\n"
            "Return ONLY valid JSON, no markdown.\n"
            "STRICT JSON FORMAT:\n"
            "{\n"
            '  \"is_barrie_competitor\": true or false,\n'
            '  \"services\": [\n'
            '    {\"raw_name\": \"...\", \"normalized_name\": \"...\", \"category\": \"...\", \"price\": 123.0, \"currency\": \"CAD\"},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            f"Text:\n{content[:4000]}"
        )

        raw = ""
        try:
            raw = self.llm.generate_text(
                prompt=prompt,
                max_tokens=350,
                temperature=0.2,
                json_mode=True,
            )
        except Exception as e:
            log.warning("LLM analysis failed: %s", e)
            return {"raw": raw, "parsed": None, "is_barrie_competitor": False, "prices": {}}

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
        prices_dict: Dict[str, List[str]] = {}

        if parsed:
            # ---- 1) base flag from LLM ----
            is_barrie_val = parsed.get("is_barrie_competitor")
            if isinstance(is_barrie_val, str):
                is_barrie = is_barrie_val.strip().lower() in {"true", "yes", "y"}
            else:
                is_barrie = bool(is_barrie_val)

            services = parsed.get("services") or []

            # Heuristic upgrades: trusted domains or clear Barrie + services signal
            domain = self._normalized_domain(url or "")
            if domain in ALWAYS_COMPETITOR_DOMAINS:
                is_barrie = True
            elif not is_barrie and services and self._looks_like_barrie_service(content, url or ""):
                is_barrie = True

            # ---- 3) Build normalized price dict ----
            if isinstance(services, list):
                for svc in services:
                    if not isinstance(svc, dict):
                        continue
                    raw_name = (svc.get("raw_name") or "").strip()
                    normalized = (svc.get("normalized_name") or "").strip().lower()
                    category = (svc.get("category") or "").strip().lower()
                    price = svc.get("price")
                    currency = (svc.get("currency") or "").strip().upper() or "CAD"

                    key = normalized or category or "other"
                    entry = raw_name or key
                    if price is not None:
                        entry = f"{raw_name or key} - {currency}{price}"

                    prices_dict.setdefault(key, []).append(entry)

        # Fallback: if no structured prices, use regex extraction on content
        if not prices_dict:
            regex_hits = self._extract_prices(content)
            if regex_hits:
                prices_dict["other"] = regex_hits

        return {
            "raw": raw,
            "parsed": parsed,
            "is_barrie_competitor": is_barrie,
            "prices": prices_dict,
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
        numeric_prices: List[float] = []

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
                        bits = []
                        for k, v in prices.items():
                            if isinstance(v, list):
                                joined = "; ".join(str(x) for x in v if x)
                            else:
                                joined = str(v)
                            bits.append(f"{k}: {joined}")
                            numeric_prices.extend(self._collect_numeric_prices({k: v}))
                        if bits:
                            price_str = "; ".join(bits)
                    elif isinstance(prices, list):
                        bits = [str(p) for p in prices if p]
                        if bits:
                            price_str = "; ".join(bits)
                        numeric_prices.extend(self._collect_numeric_prices(prices))
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

        if numeric_prices:
            avg_price = sum(numeric_prices) / len(numeric_prices)
            lines.append("")
            lines.append(f"Average service price (all captured): ${avg_price:.2f}")

        return "\n".join(lines)


if __name__ == "__main__":
    researcher = MarketResearcher()
    data = researcher.discover_competitors()
    print(researcher.format_report(data))
