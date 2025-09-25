"""Run the demo.html UI as a local web app."""

from __future__ import annotations

import os
import socket
import threading
import webbrowser
from pathlib import Path

from flask import Flask, Response


HTML_PATH = Path(__file__).parent / "demo.html"
if not HTML_PATH.exists():
    raise FileNotFoundError("demo.html が見つかりません。リポジトリ直下に配置してください。")


def load_html() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def find_free_port(default: int = 7865) -> int:
    env_port = os.environ.get("DEMO_HTML_PORT") or os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        return int(env_port)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", default))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


app = Flask(__name__)


@app.get("/")
def index() -> Response:
    return Response(load_html(), mimetype="text/html")


def open_browser(url: str) -> None:
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()


def main() -> None:
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    open_browser(url)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
