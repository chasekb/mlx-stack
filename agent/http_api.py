from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler
from typing import Mapping, cast
from urllib.parse import parse_qs, urlparse

from agent.contracts import (
    AgentHttpContext,
    AgentHttpServiceContext,
    build_agent_http_service_context,
    validate_agent_http_context,
)
from agent.http_service import (
    build_agent_run_response,
    build_metrics_response,
    build_retrieve_response,
    build_run_response,
)


def build_handler(context: Mapping[str, object]):
    validate_agent_http_context(context)
    context = cast(AgentHttpContext, context)
    service_context: AgentHttpServiceContext = build_agent_http_service_context(context)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, payload, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path == "/metrics":
                response_payload = build_metrics_response(service_context)
                self._reply(response_payload)
                return

            if parsed.path == "/tools":
                self._reply({"ok": True, "service": "agent", "tools": context["tool_schemas"]})
                return

            if parsed.path.startswith("/runs/"):
                run_id = parsed.path.split("/runs/", 1)[1].strip()
                response_payload, status = build_run_response(run_id, service_context)
                self._reply(response_payload, status=status)
                return

            if parsed.path == "/retrieve":
                qs = parse_qs(parsed.query)
                query = (qs.get("q", [""])[0] or "").strip()
                try:
                    top_k = int((qs.get("top_k", ["5"])[0] or "5"))
                except ValueError:
                    top_k = 5
                path_prefix = (qs.get("path_prefix", [""])[0] or "").strip() or None
                response_payload, status = build_retrieve_response(query, top_k, path_prefix, service_context)
                self._reply(response_payload, status=status)
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

            response_payload = build_agent_run_response(
                payload,
                context=service_context,
                perf_counter_fn=time.perf_counter,
            )
            self._reply(response_payload)

    return Handler
