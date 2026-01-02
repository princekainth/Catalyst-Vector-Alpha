import os
import time
from typing import Any

import jwt
import requests
from fastapi import Header, HTTPException

from app.core.config import settings

_JWKS_URL = "https://api.clerk.com/v1/jwks"
_JWKS_CACHE: dict[str, Any] | None = None
_JWKS_CACHE_TS = 0.0
_JWKS_TTL_SECONDS = 600
_HTTP = requests.Session()


def _get_jwks() -> dict[str, Any]:
    global _JWKS_CACHE, _JWKS_CACHE_TS
    now = time.time()
    if _JWKS_CACHE and (now - _JWKS_CACHE_TS) < _JWKS_TTL_SECONDS:
        return _JWKS_CACHE
    secret = os.getenv("CLERK_SECRET_KEY", "")
    headers = {"Authorization": f"Bearer {secret}"} if secret else None
    response = _HTTP.get(_JWKS_URL, timeout=5, headers=headers)
    response.raise_for_status()
    _JWKS_CACHE = response.json()
    _JWKS_CACHE_TS = now
    return _JWKS_CACHE


def verify_token(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        jwks = _get_jwks()
        keys = jwks.get("keys", []) if isinstance(jwks, dict) else []
        key_data = next((key for key in keys if key.get("kid") == kid), None)
        if not key_data:
            raise HTTPException(status_code=401, detail="Invalid token")

        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key_data)
        decode_kwargs: dict[str, Any] = {"algorithms": ["RS256"]}
        if settings.clerk_issuer:
            decode_kwargs["issuer"] = settings.clerk_issuer
        if settings.clerk_audience:
            decode_kwargs["audience"] = settings.clerk_audience

        payload = jwt.decode(token, public_key, **decode_kwargs)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_org_id(x_org_id: str | None = Header(default=None)) -> str:
    return x_org_id or "demo-org"
