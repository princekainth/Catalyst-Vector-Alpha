
import hashlib
import json
from typing import Any, Mapping

REDACT_KEYS = {
    "token", "password", "passwd", "secret", "key", "api_key", "apikey",
    "auth", "authorization", "bearer", "credentials",
    "env_value", "value", "env_val",
}

def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k)
            if key.lower() in REDACT_KEYS:
                out[key] = "***REDACTED***"
            else:
                out[key] = _redact(v)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value

def hash_args(args: Mapping[str, Any] | None) -> str:
    safe_args = _redact(dict(args or {}))
    canonical = json.dumps(safe_args, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def test_redaction():
    print("--- Testing Redaction Logic ---")
    
    args1 = {"tool": "k8s_patch_deployment_env", "env_value": "SECRET_A", "deployment": "dep1"}
    args2 = {"tool": "k8s_patch_deployment_env", "env_value": "SECRET_B", "deployment": "dep1"}
    
    hash1 = hash_args(args1)
    hash2 = hash_args(args2)
    
    print(f"Hash 1 (SECRET_A): {hash1}")
    print(f"Hash 2 (SECRET_B): {hash2}")
    
    if hash1 == hash2:
        print("✓ Success: Logically identical actions with different secrets produce the SAME hash (Redaction confirmed).")
    else:
        print("FAIL: Hashes differ. Secrets might be leaking into the hash calculation.")

    # Nested check
    args3 = {"nested": {"value": "SECRET_C"}}
    args4 = {"nested": {"value": "SECRET_D"}}
    if hash_args(args3) == hash_args(args4):
        print("✓ Success: Nested redaction confirmed.")
    else:
        print("FAIL: Nested redaction failed.")

if __name__ == "__main__":
    test_redaction()
