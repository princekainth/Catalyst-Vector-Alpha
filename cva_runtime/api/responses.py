from __future__ import annotations

from typing import Any, Optional
from flask import jsonify


def ok(data: Any, *, runtime_state: Optional[str] = None, http_status: int = 200):
    payload = {
        "status": "ok",
        "data": data,
    }
    if runtime_state is not None:
        payload["runtime_state"] = runtime_state
    return jsonify(payload), http_status


def err(code: str, message: str, *, details: Any = None, http_status: int = 400):
    payload = {
        "status": "error",
        "error": {
            "code": code,
            "message": message,
        },
    }
    if details is not None:
        payload["error"]["details"] = details
    return jsonify(payload), http_status
