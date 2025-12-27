"""URL Cache - SQLite-backed caching for web responses."""
import sqlite3, json, hashlib, time
from pathlib import Path
from typing import Optional

class URLCache:
    def __init__(self, db_path: str = "persistence_data/browser_cache.db", ttl_seconds: int = 3600):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS url_cache (
                url_hash TEXT PRIMARY KEY, url TEXT, content TEXT,
                content_type TEXT, status_code INTEGER, timestamp REAL, metadata TEXT)""")
            conn.commit()
    
    def _hash_url(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:32]
    
    def get(self, url: str) -> Optional[dict]:
        url_hash = self._hash_url(url)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT content, content_type, status_code, timestamp FROM url_cache WHERE url_hash = ?", (url_hash,)).fetchone()
        if not row or time.time() - row[3] > self.ttl:
            return None
        return {"url": url, "content": row[0], "content_type": row[1], "status_code": row[2], "from_cache": True}
    
    def set(self, url: str, content: str, content_type: str = "text/html", status_code: int = 200):
        url_hash = self._hash_url(url)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO url_cache (url_hash, url, content, content_type, status_code, timestamp) VALUES (?,?,?,?,?,?)",
                (url_hash, url, content, content_type, status_code, time.time()))
            conn.commit()
    
    def clear_expired(self):
        cutoff = time.time() - self.ttl
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM url_cache WHERE timestamp < ?", (cutoff,))
            conn.commit()
