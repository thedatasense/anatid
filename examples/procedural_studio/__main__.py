"""Serve the visual locally, or export a standalone HTML file. No extra dependencies."""

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading

from .data import build_demo
from .router import DEFAULT_MODEL, OpenRouterGuide, RouterError, guidance_context, load_key

ROOT = Path(__file__).resolve().parent
BRAND = ROOT.parents[1] / "assets" / "brand"
ASSETS = {
    "/": (ROOT / "index.html", "text/html"),
    "/style.css": (ROOT / "style.css", "text/css"),
    "/app.js": (ROOT / "app.js", "text/javascript"),
    "/brand/anatid-logo.svg": (BRAND / "anatid-logo.svg", "image/svg+xml"),
    "/brand/favicon.svg": (BRAND / "favicon.svg", "image/svg+xml"),
    "/favicon.ico": (BRAND / "favicon.ico", "image/x-icon"),
}


def standalone(payload: dict) -> str:
    """Inline only local assets and escape JSON so the exported page is safe to embed."""
    payload = {**payload, "llm": {"enabled": False, "model": None}}
    data = json.dumps(payload).replace("<", "\\u003c").replace("&", "\\u0026")
    html = (
        ROOT.joinpath("index.html")
        .read_text()
        .replace(
            '<link rel="stylesheet" href="/style.css">',
            "<style>" + ROOT.joinpath("style.css").read_text() + "</style>",
        )
        .replace(
            '<script src="/app.js" defer></script>',
            '<script id="demo-data" type="application/json">'
            + data
            + "</script>"
            + "<script defer>"
            + ROOT.joinpath("app.js").read_text()
            + "</script>",
        )
    )
    for name in ("anatid-logo.svg", "favicon.svg"):
        encoded = base64.b64encode(BRAND.joinpath(name).read_bytes()).decode("ascii")
        html = html.replace(f"/brand/{name}", f"data:image/svg+xml;base64,{encoded}")
    return html


def handler_for(payload: dict, guide: OpenRouterGuide | None = None):
    """Serve the replay and an optional bounded guidance call. Never execute model actions."""
    token = secrets.token_urlsafe(24)
    exposed = {
        **payload,
        "llm": {
            "enabled": guide is not None,
            "model": guide.model if guide else None,
            "token": token if guide else None,
        },
    }
    encoded = json.dumps(exposed).encode()
    model_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            content = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def local_host(self):
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def do_GET(self):
            if not self.local_host():
                self.send_error(403)
                return
            if self.path == "/api/demo":
                content, kind = encoded, "application/json"
            elif self.path in ASSETS:
                filename, kind = ASSETS[self.path]
                content = filename.read_bytes()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if self.path != "/api/guide":
                self.send_json(404, {"error": "Unknown endpoint."})
                return
            origin = self.headers.get("Origin")
            if (
                not self.local_host()
                or (origin and origin != f"http://{self.headers.get('Host')}")
                or not secrets.compare_digest(self.headers.get("X-Demo-Token", ""), token)
            ):
                self.send_json(403, {"error": "This request must originate from the local demo."})
                return
            if guide is None:
                self.send_json(409, {"error": "Start the server with --live to enable OpenRouter."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected a small JSON request.")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object.")
                context = guidance_context(
                    payload, body.get("case_id"), body.get("checkpoint"), body.get("step")
                )
            except (ValueError, TypeError):
                self.send_json(400, {"error": "Invalid case, checkpoint, or replay step."})
                return
            if not model_lock.acquire(blocking=False):
                self.send_json(
                    429, {"error": "A model request is already running. Wait for it to finish."}
                )
                return
            try:
                self.send_json(200, guide.guide(context))
            except RouterError as exc:
                self.send_json(502, {"error": str(exc)})
            finally:
                model_lock.release()

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable server-side OpenRouter guidance using your configured key.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="OpenRouter model ID; default routes automatically."
    )
    parser.add_argument(
        "--export", type=Path, help="Write a new, standalone HTML file instead of serving."
    )
    args = parser.parse_args()
    if args.export and args.live:
        parser.error("--export is offline; use --live with the local server")
    payload = build_demo()
    if args.export:
        try:
            with args.export.open("x", encoding="utf-8") as output:
                output.write(standalone(payload))
        except FileExistsError:
            parser.error("--export must name a new file")
        print(f"Exported {args.export}")
        return
    guide = None
    if args.live:
        key = load_key()
        if not key:
            parser.error(
                "Set OPEN_ROUTER_KEY or OPENROUTER_API_KEY, or open_router_key= in the repository .env"
            )
        guide = OpenRouterGuide(key, args.model)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(payload, guide)) as server:
        print(f"Procedural studio: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
