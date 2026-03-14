from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler
from typing import Mapping, cast
from urllib.parse import parse_qs, urlparse

from agent.contracts import AgentHttpContext, validate_agent_http_context
from agent.http_service import build_agent_run_response


def build_handler(context: Mapping[str, object]):
    validate_agent_http_context(context)
    context = cast(AgentHttpContext, context)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, payload, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path == "/metrics":
                metrics = context["load_metrics"]()
                kv_obj = context["load_kv_cache"]()
                kv_summary = {}
                for model_name, store in kv_obj.get("models", {}).items():
                    if not isinstance(store, dict):
                        continue
                    entries = store.get("entries", {}) if isinstance(store.get("entries", {}), dict) else {}
                    kv_summary[model_name] = {
                        "entries": len(entries),
                        "used_tokens": int(store.get("used_tokens", 0)),
                        "budget_tokens": int(
                            store.get("budget_tokens", context["default_kv_model_budget_tokens"])
                        ),
                    }
                thresholds = context["parse_alert_thresholds"]()
                alerts = context["compute_alerts"](metrics, thresholds=thresholds)
                if alerts:
                    context["emit_event"]("alerts_emitted", alerts=alerts)
                self._reply(
                    {
                        "ok": True,
                        "service": "agent",
                        "metrics": metrics,
                        "kv_cache": {"models": kv_summary},
                        "alerts": alerts,
                        "alert_thresholds": thresholds,
                    }
                )
                return

            if parsed.path == "/tools":
                self._reply({"ok": True, "service": "agent", "tools": context["tool_schemas"]})
                return

            if parsed.path.startswith("/runs/"):
                run_id = parsed.path.split("/runs/", 1)[1].strip()
                target = (context["runs_dir"] / f"{run_id}.json").resolve()
                if not target.exists() or not target.is_file() or not context["ensure_under_root"](target):
                    self._reply({"error": "run_not_found"}, status=404)
                    return
                payload = json.loads(target.read_text(encoding="utf-8"))
                self._reply({"ok": True, "service": "agent", "run": payload})
                return

            if parsed.path == "/retrieve":
                if not context["index_path"].exists():
                    self._reply({"error": "missing_index", "detail": "Run `ai-dev index .` first."}, status=400)
                    return

                qs = parse_qs(parsed.query)
                query = (qs.get("q", [""])[0] or "").strip()
                if not query:
                    self._reply({"error": "missing_query", "detail": "Provide q=<query>"}, status=400)
                    return

                try:
                    top_k = int((qs.get("top_k", ["5"])[0] or "5"))
                except ValueError:
                    top_k = 5
                top_k = max(1, min(top_k, 20))
                path_prefix = (qs.get("path_prefix", [""])[0] or "").strip() or None

                index_obj = json.loads(context["index_path"].read_text(encoding="utf-8"))
                payload = context["retrieve"](index_obj, query=query, top_k=top_k, path_prefix=path_prefix)
                self._reply({"ok": True, "service": "agent", "retrieval": payload})
                return

            if parsed.path == "/health":
                self._reply({"ok": True, "service": "agent"})
                return

            self._reply({"error": "not found"}, status=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/agent/run":
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

            response_payload = build_agent_run_response(payload, context=context, perf_counter_fn=time.perf_counter)
            self._reply(response_payload)

    return Handler
