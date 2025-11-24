"""
Shared alert store for inter-agent communication.
Prevents alerts from being lost due to memory compression.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

class AlertStore:
    """Persistent alert storage that survives memory compression."""
    
    def __init__(self, filepath: str = "persistence_data/alerts.json"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create file if it doesn't exist."""
        if not self.filepath.exists():
            self.filepath.write_text("[]")
    
    def add_alert(self, alert: Dict) -> None:
        """Add an alert to the store."""
        try:
            alerts = self.get_all_alerts()
            alert["timestamp"] = alert.get("timestamp", time.time())
            alerts.append(alert)
            
            # Keep only last 100 alerts
            alerts = alerts[-100:]
            
            self.filepath.write_text(json.dumps(alerts, indent=2))
            print(f"[AlertStore] Added alert: {alert.get('type')} -> {alert.get('target_deployment')}")
        except Exception as e:
            print(f"[AlertStore ERROR] Failed to add alert: {e}")
    
    def get_all_alerts(self) -> List[Dict]:
        """Get all alerts."""
        try:
            return json.loads(self.filepath.read_text())
        except Exception:
            return []
    
    def get_recent_alerts(self, alert_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get recent alerts, optionally filtered by type."""
        alerts = self.get_all_alerts()
        
        if alert_type:
            alerts = [a for a in alerts if a.get("type") == alert_type]
        
        # Return most recent first
        return list(reversed(alerts[-limit:]))
    
    def clear_old_alerts(self, max_age_seconds: float = 3600):
        """Remove alerts older than max_age_seconds."""
        try:
            alerts = self.get_all_alerts()
            now = time.time()
            alerts = [a for a in alerts if (now - a.get("timestamp", 0)) < max_age_seconds]
            self.filepath.write_text(json.dumps(alerts, indent=2))
        except Exception as e:
            print(f"[AlertStore ERROR] Failed to clear old alerts: {e}")

# Global singleton
_alert_store = None

def get_alert_store() -> AlertStore:
    """Get the global alert store instance."""
    global _alert_store
    if _alert_store is None:
        _alert_store = AlertStore()
    return _alert_store