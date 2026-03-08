import json
from http.server import BaseHTTPRequestHandler, HTTPServer


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

        draft_tokens = payload.get("draft_tokens", [])
        target_tokens = payload.get("target_tokens", [])
        if not isinstance(draft_tokens, list) or not isinstance(target_tokens, list):
            self._reply({"error": "invalid_tokens", "detail": "draft_tokens and target_tokens must be arrays"}, status=400)
            return

        result = run_speculative_loop(
            draft_tokens=[str(t) for t in draft_tokens],
            target_tokens=[str(t) for t in target_tokens],
        )
        self._reply({"ok": True, "service": "spec-router", "result": result})


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8092), Handler)
    print("Spec router listening on :8092")
    server.serve_forever()
