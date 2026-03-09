from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer


def tokenize_text(text: str) -> list[str]:
    return [tok for tok in re.split(r"\s+", (text or "").strip()) if tok]


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 20.0) -> dict:
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


def extract_completion_text(resp: dict) -> str:
    if not isinstance(resp, dict):
        return ""
    choices = resp.get("choices", []) if isinstance(resp.get("choices", []), list) else []
    if not choices:
        return ""
    c0 = choices[0] if isinstance(choices[0], dict) else {}
    text = c0.get("text")
    if isinstance(text, str) and text.strip():
        return text
    msg = c0.get("message") if isinstance(c0.get("message"), dict) else {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    return ""


def request_model_tokens(
    *,
    api_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[list[str], float]:
    started = time.perf_counter()
    resp = http_json(
        "POST",
        api_url.rstrip("/"),
        payload={
            "model": model,
            "prompt": prompt,
            "max_tokens": max(1, int(max_tokens)),
            "temperature": 0,
        },
        timeout=timeout,
    )
    text = extract_completion_text(resp)
    took_ms = (time.perf_counter() - started) * 1000.0
    return tokenize_text(text), took_ms


def _first_token(tokens: list[str]) -> str:
    if not tokens:
        return ""
    return str(tokens[0])


def run_streaming_acceptance_loop(
    *,
    prompt: str,
    draft_model: str,
    target_model: str,
    draft_url: str,
    target_url: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    accepted = 0
    compared = 0
    rejected = 0
    output_tokens: list[str] = []
    draft_errors: list[str] = []
    step_trace: list[dict] = []
    draft_ms_total = 0.0
    target_ms_total = 0.0

    for step in range(max(1, int(max_tokens))):
        running_prompt = (prompt + " " + " ".join(output_tokens)).strip()

        draft_token = ""
        draft_step_ms = 0.0
        try:
            draft_tokens, draft_step_ms = request_model_tokens(
                api_url=draft_url,
                model=draft_model,
                prompt=running_prompt,
                max_tokens=1,
                timeout=timeout,
            )
            draft_token = _first_token(draft_tokens)
        except Exception as e:
            draft_errors.append(str(e))

        target_tokens, target_step_ms = request_model_tokens(
            api_url=target_url,
            model=target_model,
            prompt=running_prompt,
            max_tokens=1,
            timeout=timeout,
        )
        target_token = _first_token(target_tokens)

        draft_ms_total += draft_step_ms
        target_ms_total += target_step_ms

        if not target_token:
            break

        compared += 1
        matched = bool(draft_token) and draft_token == target_token
        if matched:
            accepted += 1
        else:
            rejected += 1

        output_tokens.append(target_token)
        step_trace.append(
            {
                "step": step + 1,
                "draft": draft_token,
                "target": target_token,
                "accepted": matched,
            }
        )

    acceptance_rate = (accepted / compared) if compared else 0.0
    out = {
        "accepted_tokens": accepted,
        "compared_tokens": compared,
        "rejected_tokens": rejected,
        "acceptance_rate": round(acceptance_rate, 4),
        "output_tokens": output_tokens,
        "draft_token_count": compared,
        "target_token_count": compared,
        "draft_call_ms": round(draft_ms_total, 2),
        "target_call_ms": round(target_ms_total, 2),
        "loop_mode": "stream_acceptance",
        "steps": step_trace,
    }
    if draft_errors:
        out["draft_error"] = "; ".join(draft_errors)[:1000]
    return out


def run_speculative_loop(draft_tokens: list[str], target_tokens: list[str]) -> dict:
    accepted = 0
    compared = min(len(draft_tokens), len(target_tokens))
    out_tokens: list[str] = []

    for i in range(compared):
        d = draft_tokens[i]
        t = target_tokens[i]
        if d == t:
            accepted += 1
            out_tokens.append(d)
        else:
            out_tokens.append(t)

    if len(target_tokens) > compared:
        out_tokens.extend(target_tokens[compared:])

    acceptance_rate = (accepted / compared) if compared else 0.0
    return {
        "accepted_tokens": accepted,
        "compared_tokens": compared,
        "acceptance_rate": round(acceptance_rate, 4),
        "output_tokens": out_tokens,
    }


def run_speculative_decode(payload: dict) -> dict:
    draft_tokens = payload.get("draft_tokens", [])
    target_tokens = payload.get("target_tokens", [])

    if isinstance(draft_tokens, list) and isinstance(target_tokens, list) and (draft_tokens or target_tokens):
        result = run_speculative_loop(
            draft_tokens=[str(t) for t in draft_tokens],
            target_tokens=[str(t) for t in target_tokens],
        )
        result["source"] = "provided_tokens"
        return result

    prompt = str(payload.get("prompt", "") or "").strip()
    if not prompt:
        raise ValueError("missing_prompt_or_tokens")

    draft_model = str(payload.get("draft_model") or os.environ.get("SPEC_DRAFT_MODEL", "local-mlx-fast"))
    target_model = str(payload.get("target_model") or os.environ.get("SPEC_TARGET_MODEL", "local-mlx"))
    draft_url = str(
        payload.get("draft_url") or os.environ.get("SPEC_DRAFT_URL", "http://localhost:4000/v1/completions")
    )
    target_url = str(
        payload.get("target_url") or os.environ.get("SPEC_TARGET_URL", "http://localhost:4000/v1/completions")
    )
    max_tokens = int(payload.get("max_tokens", 128) or 128)
    timeout = float(payload.get("timeout", 20.0) or 20.0)
    stream_loop = bool(payload.get("stream_loop", True))

    if stream_loop:
        result = run_streaming_acceptance_loop(
            prompt=prompt,
            draft_model=draft_model,
            target_model=target_model,
            draft_url=draft_url,
            target_url=target_url,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        result.update(
            {
                "source": "model_calls",
                "draft_model": draft_model,
                "target_model": target_model,
            }
        )
        return result

    draft_tokens_out: list[str] = []
    target_tokens_out: list[str] = []
    draft_ms = 0.0
    target_ms = 0.0
    draft_error = ""

    try:
        draft_tokens_out, draft_ms = request_model_tokens(
            api_url=draft_url,
            model=draft_model,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as e:
        draft_error = str(e)

    target_tokens_out, target_ms = request_model_tokens(
        api_url=target_url,
        model=target_model,
        prompt=prompt,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    result = run_speculative_loop(draft_tokens=draft_tokens_out, target_tokens=target_tokens_out)
    result.update(
        {
            "source": "model_calls",
            "draft_model": draft_model,
            "target_model": target_model,
            "draft_token_count": len(draft_tokens_out),
            "target_token_count": len(target_tokens_out),
            "draft_call_ms": round(draft_ms, 2),
            "target_call_ms": round(target_ms, 2),
        }
    )
    if draft_error:
        result["draft_error"] = draft_error
    return result


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self._reply({"ok": True, "service": "spec-router"})
            return
        self._reply({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path != "/spec/decode":
            self._reply({"error": "not found"}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        body = self.rfile.read(max(0, content_length))

        try:
            payload = json.loads(body.decode("utf-8") if body else "{}")
        except Exception:
            self._reply({"error": "invalid_json"}, status=400)
            return

        try:
            result = run_speculative_decode(payload)
        except ValueError as e:
            self._reply({"error": str(e)}, status=400)
            return
        except urllib.error.URLError as e:
            self._reply({"error": "model_backend_unreachable", "detail": str(e)}, status=502)
            return
        except Exception as e:
            self._reply({"error": "decode_failed", "detail": str(e)}, status=500)
            return
        self._reply({"ok": True, "service": "spec-router", "result": result})


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8092), Handler)
    print("Spec router listening on :8092")
    server.serve_forever()
