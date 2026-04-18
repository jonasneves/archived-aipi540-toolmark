"""Local dev server for the static web app.

The deployed app is fully static (GitHub Pages) and runs inference in the
browser via ONNX Runtime Web + WebGPU. This script exists so `python app.py`
serves `web/` on http://localhost:8000 during development.
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
from functools import partial
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parent / "web"


def serve(port: int = 8000) -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB_ROOT))
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"serving {WEB_ROOT} at http://localhost:{port}")
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Toolmark dev server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(port=args.port)


if __name__ == "__main__":
    main()
