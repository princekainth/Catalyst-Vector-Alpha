"""Content Extractor - Convert HTML to clean LLM-friendly text."""
import re
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ExtractedContent:
    title: str
    text: str
    links: List[Dict[str, str]]
    code_blocks: List[str]
    word_count: int

class ContentExtractor:
    REMOVE_TAGS = ['script', 'style', 'noscript', 'iframe', 'nav', 'footer', 'header']
    BLOCK_TAGS = ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'li', 'br', 'blockquote', 'pre']
    
    def __init__(self, max_length: int = 8000):
        self.max_length = max_length
    
    def extract(self, html: str, url: str = "") -> ExtractedContent:
        title = self._extract_title(html)
        code_blocks = self._extract_code_blocks(html)
        links = self._extract_links(html)
        text = self._html_to_text(html)
        if len(text) > self.max_length:
            text = text[:self.max_length] + "... [truncated]"
        return ExtractedContent(title=title, text=text, links=links[:20], code_blocks=code_blocks[:10], word_count=len(text.split()))
    
    def _extract_title(self, html: str) -> str:
        match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        return match.group(1).strip() if match else "Untitled"
    
    def _extract_code_blocks(self, html: str) -> List[str]:
        blocks = []
        for m in re.finditer(r'<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>', html, re.DOTALL | re.IGNORECASE):
            code = re.sub(r'<[^>]+>', '', m.group(1)).strip()
            if code: blocks.append(code)
        return blocks
    
    def _extract_links(self, html: str) -> List[Dict[str, str]]:
        links = []
        for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>', html, re.IGNORECASE):
            href, text = m.groups()
            if text.strip() and not href.startswith(('#', 'javascript:', 'mailto:')):
                links.append({"url": href, "text": text.strip()[:100]})
        return links
    
    def _html_to_text(self, html: str) -> str:
        text = html
        for tag in self.REMOVE_TAGS:
            text = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        for tag in self.BLOCK_TAGS:
            text = re.sub(rf'</?{tag}[^>]*>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def summarize_for_memory(self, html: str, url: str, topic: str = "") -> dict:
        content = self.extract(html, url)
        return {
            "type": "WebKnowledge",
            "source_url": url,
            "title": content.title,
            "topic": topic,
            "text_preview": content.text[:500],
            "code_examples": content.code_blocks[:3],
            "word_count": content.word_count
        }
