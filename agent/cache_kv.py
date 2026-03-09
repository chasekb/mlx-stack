from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / ".ai-dev" / "prompt_cache.json"
KV_CACHE_PATH = ROOT / ".ai-dev" / "kv_cache.json"
DEFAULT_KV_MODEL_BUDGET_TOKENS = 8000
DEFAULT_KV_ENTRY_MAX_TOKENS = 2048


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_cache() -> dict:
    cache = load_json_file(CACHE_PATH, {"schema_version": 1, "updated_at": utc_now_iso(), "entries": {}})
    if not isinstance(cache, dict):
        return {"schema_version": 1, "updated_at": utc_now_iso(), "entries": {}}
    if not isinstance(cache.get("entries"), dict):
        cache["entries"] = {}
    return cache


def save_cache(cache_obj: dict) -> None:
    cache_obj["updated_at"] = utc_now_iso()
    save_json_file(CACHE_PATH, cache_obj)


def get_cache_entry(cache_obj: dict, key: str, namespace: str) -> dict | None:
    entry = cache_obj.get("entries", {}).get(key)
    if not isinstance(entry, dict):
        return None
    if entry.get("namespace") != namespace:
        return None
    expires_at = float(entry.get("expires_at_epoch", 0.0) or 0.0)
    now = time.time()
    if expires_at and now > expires_at:
        cache_obj.get("entries", {}).pop(key, None)
        return None
    return entry


def set_cache_entry(cache_obj: dict, key: str, namespace: str, result: dict, ttl_seconds: int) -> None:
    now = time.time()
    cache_obj.setdefault("entries", {})[key] = {
        "namespace": namespace,
        "created_at": utc_now_iso(),
        "created_at_epoch": now,
        "expires_at_epoch": now + max(1, ttl_seconds),
        "result": result,
    }


def estimate_tokens(text: str) -> int:
    return max(1, len([t for t in re.split(r"\s+", (text or "").strip()) if t]))


def normalize_prefix_text(payload: dict) -> str:
    raw = payload.get("prefix")
    if raw is None:
        raw = payload.get("prompt_prefix")
    if raw is None:
        raw = ""
    return str(raw).strip()


def hash_prefix(prefix: str) -> str:
    return hashlib.sha256((prefix or "").encode("utf-8")).hexdigest()


def load_kv_cache() -> dict:
    obj = load_json_file(KV_CACHE_PATH, {"schema_version": 1, "updated_at": utc_now_iso(), "models": {}})
    if not isinstance(obj, dict):
        obj = {"schema_version": 1, "updated_at": utc_now_iso(), "models": {}}
    if not isinstance(obj.get("models"), dict):
        obj["models"] = {}
    return obj


def save_kv_cache(obj: dict) -> None:
    obj["updated_at"] = utc_now_iso()
    save_json_file(KV_CACHE_PATH, obj)


def _ensure_model_store(kv_obj: dict, model: str, budget_tokens: int) -> dict:
    models = kv_obj.setdefault("models", {})
    store = models.get(model)
    if not isinstance(store, dict):
        store = {
            "budget_tokens": max(256, int(budget_tokens)),
            "entries": {},
            "updated_at": utc_now_iso(),
        }
        models[model] = store
    store["budget_tokens"] = max(256, int(budget_tokens))
    if not isinstance(store.get("entries"), dict):
        store["entries"] = {}
    return store


def _enforce_model_budget(store: dict, keep_key: str) -> list[str]:
    entries = store.get("entries", {}) if isinstance(store.get("entries", {}), dict) else {}
    budget = max(256, int(store.get("budget_tokens", DEFAULT_KV_MODEL_BUDGET_TOKENS)))
    evicted: list[str] = []

    def used_tokens() -> int:
        return sum(int(v.get("token_estimate", 0) or 0) for v in entries.values() if isinstance(v, dict))

    while used_tokens() > budget and entries:
        candidates = [(k, v) for k, v in entries.items() if isinstance(v, dict) and k != keep_key]
        if not candidates:
            break
        candidates.sort(key=lambda kv: float(kv[1].get("updated_at_epoch", 0.0) or 0.0))
        drop_key = candidates[0][0]
        entries.pop(drop_key, None)
        evicted.append(drop_key)

    store["entries"] = entries
    store["used_tokens"] = used_tokens()
    store["updated_at"] = utc_now_iso()
    return evicted


def get_kv_reuse_status(payload: dict) -> dict:
    kv_cfg = payload.get("kv_cache", {}) if isinstance(payload.get("kv_cache", {}), dict) else {}
    enabled = bool(kv_cfg.get("enabled", True))
    if not enabled:
        return {"enabled": False, "status": "disabled"}

    tenant_id = str(kv_cfg.get("tenant_id", "default")).strip() or "default"
    session_id = str(kv_cfg.get("session_id", "")).strip()
    if not session_id:
        return {"enabled": True, "status": "bypass", "reason": "missing_session_id", "tenant_id": tenant_id}

    model = str(payload.get("model") or kv_cfg.get("model") or "default").strip() or "default"
    prefix = normalize_prefix_text(kv_cfg)
    if not prefix:
        return {
            "enabled": True,
            "status": "bypass",
            "reason": "missing_prefix",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "model": model,
        }

    provided_hash = str(kv_cfg.get("prefix_hash", "")).strip()
    computed_hash = hash_prefix(prefix)
    if provided_hash and provided_hash != computed_hash:
        return {
            "enabled": True,
            "status": "rejected",
            "reason": "prefix_hash_mismatch",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "model": model,
        }

    max_entry_tokens = max(1, int(kv_cfg.get("entry_max_tokens", DEFAULT_KV_ENTRY_MAX_TOKENS)))
    token_estimate = min(estimate_tokens(prefix), max_entry_tokens)
    model_budget_tokens = max(256, int(kv_cfg.get("model_budget_tokens", DEFAULT_KV_MODEL_BUDGET_TOKENS)))

    kv_obj = load_kv_cache()
    model_store = _ensure_model_store(kv_obj, model=model, budget_tokens=model_budget_tokens)
    entries = model_store.get("entries", {})
    session_key = f"{tenant_id}|{session_id}|{model}"
    prev = entries.get(session_key) if isinstance(entries, dict) else None

    reused = False
    reuse_reason = "cold_start"
    reused_tokens = 0
    if isinstance(prev, dict):
        prev_prefix = str(prev.get("prefix", ""))
        if prefix.startswith(prev_prefix):
            reused = True
            reuse_reason = "prefix_extension"
            reused_tokens = min(int(prev.get("token_estimate", 0) or 0), token_estimate)
        else:
            reuse_reason = "prefix_boundary_mismatch"

    now_epoch = time.time()
    entries[session_key] = {
        "tenant_id": tenant_id,
        "session_id": session_id,
        "model": model,
        "prefix": prefix,
        "prefix_hash": computed_hash,
        "prefix_chars": len(prefix),
        "token_estimate": token_estimate,
        "updated_at": utc_now_iso(),
        "updated_at_epoch": now_epoch,
    }
    model_store["entries"] = entries
    evicted = _enforce_model_budget(model_store, keep_key=session_key)
    save_kv_cache(kv_obj)

    return {
        "enabled": True,
        "status": "hit" if reused else "miss",
        "reason": reuse_reason,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "model": model,
        "session_key": session_key,
        "prefix_hash": computed_hash,
        "token_estimate": token_estimate,
        "reused_tokens": reused_tokens,
        "model_budget_tokens": int(model_store.get("budget_tokens", model_budget_tokens)),
        "model_used_tokens": int(model_store.get("used_tokens", 0)),
        "evicted_entries": evicted,
    }
