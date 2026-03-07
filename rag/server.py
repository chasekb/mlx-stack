from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def do_GET(self):
        if self.path == '/health':
            self._reply({'ok': True, 'service': 'rag'})
            return
        self._reply({'error': 'not found'}, status=404)


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8090), Handler)
    print('RAG service listening on :8090')
    server.serve_forever()
