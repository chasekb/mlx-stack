from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / ".ai-dev" / "index.json"


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(tok) >= 2]


def retrieve(index_obj: dict, query: str, top_k: int = 5, path_prefix: str | None = None) -> dict:
    query_terms = set(tokenize(query))
    if not query_terms:
        return {"query": query, "top_symbols": [], "top_chunks": []}

    path_prefix = path_prefix or ""

    symbol_results = []
    for s in index_obj.get("symbols", []):
        score = 0.0
        name_terms = set(tokenize(s.get("name", "")))
        score += len(query_terms.intersection(name_terms)) * 3
        score += 1 if any(t in s.get("name", "").lower() for t in query_terms) else 0
        p = s.get("path", "")
        if path_prefix and p.startswith(path_prefix):
            score += 1.5
        if score > 0:
            symbol_results.append({"score": score, **s})

    chunk_results = []
    for c in index_obj.get("chunks", []):
        score = 0.0
        chunk_terms = set(c.get("terms", []))
        score += len(query_terms.intersection(chunk_terms))
        p = c.get("path", "")
        if path_prefix and p.startswith(path_prefix):
            score += 2.0
        if score > 0:
            chunk_results.append(
                {
                    "score": score,
                    "path": p,
                    "chunk_id": c.get("chunk_id"),
                    "start_line": c.get("start_line"),
                    "end_line": c.get("end_line"),
                    "text_preview": c.get("text_preview", ""),
                }
            )

    symbol_results.sort(key=lambda x: x["score"], reverse=True)
    chunk_results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": query,
        "top_symbols": symbol_results[:top_k],
        "top_chunks": chunk_results[:top_k],
    }


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/retrieve':
            if not INDEX_PATH.exists():
                self._reply({'error': 'missing_index', 'detail': 'Run `ai-dev index .` first.'}, status=400)
                return

            qs = parse_qs(parsed.query)
            query = (qs.get('q', [''])[0] or '').strip()
            if not query:
                self._reply({'error': 'missing_query', 'detail': 'Provide q=<query>'}, status=400)
                return

            try:
                top_k = int((qs.get('top_k', ['5'])[0] or '5'))
            except ValueError:
                top_k = 5
            top_k = max(1, min(top_k, 20))
            path_prefix = (qs.get('path_prefix', [''])[0] or '').strip() or None

            index_obj = json.loads(INDEX_PATH.read_text(encoding='utf-8'))
            payload = retrieve(index_obj, query=query, top_k=top_k, path_prefix=path_prefix)
            self._reply({'ok': True, 'service': 'agent', 'retrieval': payload})
            return

        if parsed.path == '/health':
            self._reply({'ok': True, 'service': 'agent'})
            return

        self._reply({'error': 'not found'}, status=404)


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8091), Handler)
    print('Agent service listening on :8091')
    server.serve_forever()
