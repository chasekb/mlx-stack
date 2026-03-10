from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone


def tokenize_for_spec(text: str) -> list[str]:
    normalized = (text or "").replace("\n", " ").strip()
    return [t for t in normalized.split(" ") if t]


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def command_spec_decode(args) -> int:
    payload: dict = {}

    if args.prompt:
        payload = {
            "prompt": args.prompt,
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "draft_url": args.draft_url,
            "target_url": args.target_url,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
        }
    else:
        if args.draft_tokens:
            draft_tokens = [t for t in args.draft_tokens if t]
        else:
            draft_tokens = tokenize_for_spec(args.draft_text)

        if args.target_tokens:
            target_tokens = [t for t in args.target_tokens if t]
        else:
            target_tokens = tokenize_for_spec(args.target_text)

        payload = {
            "draft_tokens": draft_tokens,
            "target_tokens": target_tokens,
        }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.url.rstrip("/") + "/spec/decode",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        print(f"spec-decode request failed: {e}", file=sys.stderr)
        return 2

    try:
        parsed = json.loads(body)
    except Exception:
        print("spec-decode returned invalid JSON", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(parsed, indent=2))
        return 0

    result = parsed.get("result", {}) if isinstance(parsed, dict) else {}
    if result.get("source") == "model_calls":
        print(f"source: {result.get('source')}")
        print(f"draft_model: {result.get('draft_model', '')}")
        print(f"target_model: {result.get('target_model', '')}")
        print(f"draft_call_ms: {result.get('draft_call_ms', 0)}")
        print(f"target_call_ms: {result.get('target_call_ms', 0)}")
        if result.get("draft_error"):
            print(f"draft_error: {result.get('draft_error')}")
    print(f"accepted_tokens: {result.get('accepted_tokens', 0)}")
    print(f"compared_tokens: {result.get('compared_tokens', 0)}")
    print(f"acceptance_rate: {result.get('acceptance_rate', 0.0)}")
    print("output_tokens:")
    for tok in result.get("output_tokens", []):
        print(f"- {tok}")
    return 0


def command_embed_enqueue(args) -> int:
    metadata = {}
    if args.metadata_json:
        try:
            parsed = json.loads(args.metadata_json)
            metadata = parsed if isinstance(parsed, dict) else {}
        except Exception:
            print("Invalid --metadata-json payload", file=sys.stderr)
            return 2

    payload = {
        "kind": args.kind,
        "payload": {
            "path": args.path,
            "text": args.text,
            "metadata": metadata,
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
        },
        "max_attempts": args.max_attempts,
    }

    try:
        out = http_json(
            "POST",
            args.url.rstrip("/") + "/jobs/enqueue",
            payload=payload,
            timeout=args.timeout,
        )
    except urllib.error.URLError as e:
        print(f"embed-enqueue request failed: {e}", file=sys.stderr)
        return 2

    print(json.dumps(out, indent=2) if args.json else f"Enqueued job_id={out.get('job_id')} status={out.get('status')}")
    return 0


def command_embed_stats(args) -> int:
    try:
        out = http_json("GET", args.url.rstrip("/") + "/stats", timeout=args.timeout)
    except urllib.error.URLError as e:
        print(f"embed-stats request failed: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    stats = out.get("stats", {}) if isinstance(out, dict) else {}
    print("Embedding queue stats:")
    print(f"- queued: {stats.get('queued', 0)}")
    print(f"- retry: {stats.get('retry', 0)}")
    print(f"- in_progress: {stats.get('in_progress', 0)}")
    print(f"- done: {stats.get('done', 0)}")
    print(f"- dead_letter: {stats.get('dead_letter', 0)}")
    return 0
