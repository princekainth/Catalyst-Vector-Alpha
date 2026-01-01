import json
import os
import tempfile
import threading
import time

QUARANTINE_PATH = os.path.join("persistence_data", "quarantine.json")
_LOCK = threading.Lock()


def _read_quarantine() -> dict:
    if not os.path.exists(QUARANTINE_PATH):
        return {}
    try:
        with open(QUARANTINE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_quarantine(data: dict) -> None:
    os.makedirs(os.path.dirname(QUARANTINE_PATH), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".quarantine_", dir=os.path.dirname(QUARANTINE_PATH))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, QUARANTINE_PATH)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _prune_expired(data: dict, now: float) -> bool:
    changed = False
    for key, entry in list(data.items()):
        until_ts = entry.get("until_ts")
        if until_ts is not None and now >= float(until_ts):
            data.pop(key, None)
            changed = True
    return changed


def get_quarantine() -> dict:
    now = time.time()
    with _LOCK:
        data = _read_quarantine()
        if _prune_expired(data, now):
            _write_quarantine(data)
        return data


def is_quarantined(key: str, now: float | None = None) -> bool:
    if not key:
        return False
    now = time.time() if now is None else now
    with _LOCK:
        data = _read_quarantine()
        entry = data.get(key)
        if not entry:
            return False
        until_ts = entry.get("until_ts")
        if until_ts is not None and now >= float(until_ts):
            data.pop(key, None)
            _write_quarantine(data)
            return False
        return True


def set_quarantine(
    key: str,
    status: str,
    until_ts: float | None,
    reason: str,
    source: str,
) -> None:
    if not key:
        return
    now = time.time()
    entry = {
        "status": status,
        "until_ts": until_ts,
        "reason": reason,
        "source": source,
        "updated_ts": now,
    }
    with _LOCK:
        data = _read_quarantine()
        data[key] = entry
        _write_quarantine(data)
