"""launcher 헬퍼 함수 테스트 — webview 안 띄움."""

from __future__ import annotations

import socket
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from aurora_ict.main import (
    _is_port_free,
    _pick_port,
    _wait_ready,
)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_is_port_free_true() -> None:
    port = _free_port()
    assert _is_port_free("127.0.0.1", port) is True


def test_is_port_free_false() -> None:
    port = _free_port()
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", port))
        s.listen(1)
        assert _is_port_free("127.0.0.1", port) is False


def test_pick_port_uses_preferred_when_free() -> None:
    port = _free_port()
    chosen = _pick_port("127.0.0.1", port)
    assert chosen == port


def test_pick_port_falls_back_when_busy() -> None:
    busy = _free_port()
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", busy))
        s.listen(1)
        chosen = _pick_port("127.0.0.1", busy)
        assert chosen != busy
        assert chosen > 0


class _OkHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args, **kwargs) -> None:  # silence
        pass


@pytest.fixture
def _http_server():
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _OkHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    yield port
    server.shutdown()
    server.server_close()


def test_wait_ready_success(_http_server: int) -> None:
    url = f"http://127.0.0.1:{_http_server}/"
    assert _wait_ready(url, timeout_sec=2.0) is True


def test_wait_ready_timeout() -> None:
    # 점유 안 된 포트 — 타임아웃 검증
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    assert _wait_ready(url, timeout_sec=0.5) is False
