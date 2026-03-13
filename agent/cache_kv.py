from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / ".ai-dev" / "prompt_cache.json"
KV_CACHE_PATH = ROOT / ".ai-dev" / "kv_cache.json"
DEFAULT_KV_MODEL_BUDGET_TOKENS = 8000
DEFAULT_KV_ENTRY_MAX_TOKENS = 2048
DEFAULT_KV_BACKEND_URL = "http://localhost:4000/v1/completions"
DEFAULT_KV_BACKEND_TIMEOUT_SECONDS = 20.0


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


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(default)


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 20.0) -> dict:
    data = json.dumps(payload or {}).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def _extract_backend_kv_status(response_payload: dict) -> dict:
    if not isinstance(response_payload, dict):
        raise ValueError("backend_invalid_response")

    kv_block = response_payload.get("kv_cache")
    if isinstance(kv_block, dict) and str(kv_block.get("status", "")).strip():
        status = str(kv_block.get("status", "")).strip().lower()
        reused_tokens = max(0, _coerce_int(kv_block.get("reused_tokens", 0), 0))
        out = {
            "status": status,
            "reason": str(kv_block.get("reason", "backend_reported") or "backend_reported"),
            "reused_tokens": reused_tokens,
        }
        backend_session_id = str(kv_block.get("backend_session_id", "")).strip()
        if backend_session_id:
            out["backend_session_id"] = backend_session_id
        return out

    usage = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {}
    prompt_details = (
        usage.get("prompt_tokens_details")
        if isinstance(usage.get("prompt_tokens_details"), dict)
        else {}
    )
    cached_tokens = prompt_details.get("cached_tokens")
    if cached_tokens is None:
        raise ValueError("backend_missing_kv_signal")

    reused_tokens = max(0, _coerce_int(cached_tokens, 0))
    return {
        "status": "hit" if reused_tokens > 0 else "miss",
        "reason": "backend_cached_tokens",
        "reused_tokens": reused_tokens,
    }


def probe_backend_kv_reuse(*, kv_cfg: dict, model: str, prefix: str, prefix_hash: str) -> dict:
    backend_url = (
        str(kv_cfg.get("backend_url", "")).strip()
        or str(os.environ.get("AGENT_KV_BACKEND_URL", "")).strip()
        or DEFAULT_KV_BACKEND_URL
    )
    timeout = float(
        kv_cfg.get(
            "backend_timeout_seconds",
            os.environ.get("AGENT_KV_BACKEND_TIMEOUT_SECONDS", DEFAULT_KV_BACKEND_TIMEOUT_SECONDS),
        )
        or DEFAULT_KV_BACKEND_TIMEOUT_SECONDS
    )
    timeout = max(0.5, min(timeout, 120.0))

    tenant_id = str(kv_cfg.get("tenant_id", "default")).strip() or "default"
    session_id = str(kv_cfg.get("session_id", "")).strip()

    payload = {
        "model": model,
        "prompt": prefix,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "kv_cache": {
            "enabled": True,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "prefix": prefix,
            "prefix_hash": prefix_hash,
        },
    }

    for passthrough_key in ("model_budget_tokens", "entry_max_tokens"):
        if passthrough_key in kv_cfg:
            payload["kv_cache"][passthrough_key] = kv_cfg.get(passthrough_key)

    started = time.perf_counter()
    response_payload = _http_json("POST", backend_url.rstrip("/"), payload=payload, timeout=timeout)
    latency_ms = (time.perf_counter() - started) * 1000.0

    status_payload = _extract_backend_kv_status(response_payload)
    status_payload["backend_url"] = backend_url
    status_payload["backend_latency_ms"] = round(latency_ms, 2)
    return status_payload


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

    session_key = f"{tenant_id}|{session_id}|{model}"
    token_estimate = estimate_tokens(prefix)

    try:
        backend_status = probe_backend_kv_reuse(
            kv_cfg=kv_cfg,
            model=model,
            prefix=prefix,
            prefix_hash=computed_hash,
        )
    except (urllib.error.URLError, TimeoutError) as e:
        return {
            "enabled": True,
            "status": "error",
            "reason": "backend_unreachable",
            "detail": str(e)[:500],
            "tenant_id": tenant_id,
            "session_id": session_id,
            "model": model,
            "session_key": session_key,
            "prefix_hash": computed_hash,
            "token_estimate": token_estimate,
            "reused_tokens": 0,
            "source": "backend",
        }
    except Exception as e:
        return {
            "enabled": True,
            "status": "error",
            "reason": "backend_probe_failed",
            "detail": str(e)[:500],
            "tenant_id": tenant_id,
            "session_id": session_id,
            "model": model,
            "session_key": session_key,
            "prefix_hash": computed_hash,
            "token_estimate": token_estimate,
            "reused_tokens": 0,
            "source": "backend",
        }

    reused_tokens = max(0, _coerce_int(backend_status.get("reused_tokens", 0), 0))
    backend_reason = str(backend_status.get("reason", "backend_reported") or "backend_reported")
    backend_state = str(backend_status.get("status", "error") or "error").strip().lower()
    if backend_state not in {"hit", "miss", "bypass", "rejected", "disabled"}:
        backend_state = "error"

    kv_obj = load_kv_cache()
    model_budget_tokens = max(256, int(kv_cfg.get("model_budget_tokens", DEFAULT_KV_MODEL_BUDGET_TOKENS)))
    model_store = _ensure_model_store(kv_obj, model=model, budget_tokens=model_budget_tokens)
    entries = model_store.get("entries", {}) if isinstance(model_store.get("entries", {}), dict) else {}
    now_epoch = time.time()
    entries[session_key] = {
        "tenant_id": tenant_id,
        "session_id": session_id,
        "model": model,
        "prefix_hash": computed_hash,
        "prefix_chars": len(prefix),
        "token_estimate": token_estimate,
        "reused_tokens": reused_tokens,
        "status": backend_state,
        "reason": backend_reason,
        "source": "backend",
        "updated_at": utc_now_iso(),
        "updated_at_epoch": now_epoch,
    }
    model_store["entries"] = entries
    model_store["used_tokens"] = sum(
        max(0, _coerce_int(v.get("reused_tokens", 0), 0))
        for v in entries.values()
        if isinstance(v, dict)
    )
    model_store["updated_at"] = utc_now_iso()
    save_kv_cache(kv_obj)

    result = {
        "enabled": True,
        "status": backend_state,
        "reason": backend_reason,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "model": model,
        "session_key": session_key,
        "prefix_hash": computed_hash,
        "token_estimate": token_estimate,
        "reused_tokens": reused_tokens,
        "model_budget_tokens": int(model_store.get("budget_tokens", model_budget_tokens)),
        "model_used_tokens": int(model_store.get("used_tokens", 0)),
        "evicted_entries": [],
        "source": "backend",
    }

    backend_url = str(backend_status.get("backend_url", "")).strip()
    if backend_url:
        result["backend_url"] = backend_url
    if "backend_latency_ms" in backend_status:
        result["backend_latency_ms"] = backend_status.get("backend_latency_ms")
    backend_session_id = str(backend_status.get("backend_session_id", "")).strip()
    if backend_session_id:
        result["backend_session_id"] = backend_session_id
    return result
