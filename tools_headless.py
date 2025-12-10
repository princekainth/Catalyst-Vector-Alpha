from typing import Any, Dict, List
from playwright.sync_api import sync_playwright


def fetch_page_content(url: str, timeout_ms: int = 20000) -> Dict[str, Any]:
    """
    Fetch page content and links using Playwright headless Chromium.
    Returns dict with HTML content and extracted links [{text, href}].
    """
    links: List[Dict[str, str]] = []
    html = ""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        html = page.content() or ""
        link_data = page.eval_on_selector_all(
            "a",
            "els => els.map(a => ({ href: a.href || '', text: (a.innerText || a.textContent || '').trim() }))",
        )
        if isinstance(link_data, list):
            links = [
                {"href": l.get("href", ""), "text": l.get("text", "")}
                for l in link_data
                if isinstance(l, dict)
            ]
        browser.close()
    return {"content": html, "links": links}
