"""
src/ui/server.py

The console's web server. Loopback only, standard library only.

    GET  /               the page (static/console.html, with a session token)
    GET  /events         server-sent events: history, then live records/poses
    GET  /api/state      theme, sticky cues, peripherals
    POST /api/say        {"text": ...}           -- the user typed to ASH
    POST /api/control    {"cmd": ..., "arg": ...} -- pause, mute, face, ...
    POST /api/theme      {"theme": ...}

Why SSE and not a websocket: the stream is one-way (ASH -> window), input is
a handful of POSTs, and SSE reconnects by itself with Last-Event-ID, which is
exactly the "reopen the window and see the conversation" behaviour wanted.

Two checks keep another program, or a web page in your browser, from driving
ASH through this port: the Host header must be loopback (defeats DNS
rebinding), and every POST must carry the token that is only ever written
into the page this server itself serves (a cross-site page cannot read it).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("ash.ui.server")

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
THEMES = ("nebula", "blossom", "paper", "forest")
_LOOPBACK = ("127.0.0.1", "localhost", "[::1]")


class ConsoleServer:
    def __init__(self, bus, host: str = "127.0.0.1", port: int = 8788,
                 theme: str = "nebula", prefs_path: Optional[str] = None,
                 on_say: Optional[Callable[[str], Any]] = None,
                 on_control: Optional[Callable[[str, str], Dict[str, Any]]] = None,
                 state_extra: Optional[Callable[[], Dict[str, Any]]] = None):
        self.bus = bus
        self.host, self.port = host, port
        self.token = secrets.token_urlsafe(18)
        self.boot = secrets.token_hex(4)
        self.prefs_path = prefs_path
        self.theme = theme if theme in THEMES else "nebula"
        self._load_prefs()
        self.on_say = on_say
        self.on_control = on_control
        self.state_extra = state_extra
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # ------------------------------------------------------------------
    def start(self) -> bool:
        server = self

        class Handler(_Handler):
            srv = server

        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError:
            logger.exception("Console could not bind %s:%s", self.host, self.port)
            return False
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        kwargs={"poll_interval": 0.5},
                                        daemon=True, name="ash-console")
        self._thread.start()
        logger.info("Console on %s", self.url)
        return True

    def stop(self):
        self._stop.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _load_prefs(self):
        if not self.prefs_path or not os.path.exists(self.prefs_path):
            return
        try:
            with open(self.prefs_path, encoding="utf-8") as f:
                t = json.load(f).get("theme")
            if t in THEMES:
                self.theme = t
        except Exception:
            logger.warning("Ignoring unreadable console prefs at %s", self.prefs_path)

    def set_theme(self, theme: str) -> bool:
        if theme not in THEMES:
            return False
        self.theme = theme
        if self.prefs_path:
            try:
                os.makedirs(os.path.dirname(self.prefs_path) or ".", exist_ok=True)
                with open(self.prefs_path, "w", encoding="utf-8") as f:
                    json.dump({"theme": theme}, f)
            except Exception:
                logger.warning("Could not save console prefs", exc_info=True)
        self.bus.publish("theme", theme=theme)
        return True

    def state(self) -> Dict[str, Any]:
        out = {"theme": self.theme, "themes": list(THEMES), "cues": dict(self.bus.state)}
        if self.state_extra is not None:
            try:
                out.update(self.state_extra() or {})
            except Exception:
                logger.debug("state_extra failed", exc_info=True)
        return out

    def page(self) -> bytes:
        with open(os.path.join(STATIC, "console.html"), encoding="utf-8") as f:
            html = f.read()
        return (html.replace("{{TOKEN}}", self.token)
                    .replace("{{THEME}}", self.theme)).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    srv: ConsoleServer = None   # set per server by subclassing
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):   # keep the daemon log for the daemon
        logger.debug("console %s", fmt % args)

    # ------------------------------------------------------------------
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in _LOOPBACK

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Dict[str, Any]):
        self._send(code, json.dumps(obj, default=str).encode("utf-8"))

    # ------------------------------------------------------------------
    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {"ok": False, "error": "loopback only"})
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send(200, self.srv.page(), "text/html; charset=utf-8")
        if path == "/events":
            return self._events()
        if path == "/api/state":
            return self._json(200, {"ok": True, **self.srv.state()})
        return self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if not self._host_ok():
            return self._json(403, {"ok": False, "error": "loopback only"})
        if not secrets.compare_digest(self.headers.get("X-ASH-Token", ""), self.srv.token):
            return self._json(403, {"ok": False, "error": "bad token"})
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 65536)
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"ok": False, "error": "bad json"})

        path = self.path.split("?", 1)[0]
        if path == "/api/say":
            text = str(body.get("text", "")).strip()[:4000]
            if not text:
                return self._json(400, {"ok": False, "error": "empty"})
            if self.srv.on_say is None:
                return self._json(503, {"ok": False, "error": "not connected to ASH"})
            self.srv.on_say(text)
            return self._json(200, {"ok": True})
        if path == "/api/control":
            if self.srv.on_control is None:
                return self._json(503, {"ok": False, "error": "not connected to ASH"})
            try:
                res = self.srv.on_control(str(body.get("cmd", "")), str(body.get("arg", "")))
            except Exception as e:
                logger.exception("Console control failed")
                res = {"ok": False, "error": str(e)}
            return self._json(200, res or {"ok": True})
        if path == "/api/theme":
            ok = self.srv.set_theme(str(body.get("theme", "")))
            return self._json(200 if ok else 400, {"ok": ok, "theme": self.srv.theme})
        return self._json(404, {"ok": False, "error": "not found"})

    # ------------------------------------------------------------------
    def _events(self):
        # The whole history every time, ignoring Last-Event-ID: after a daemon
        # restart the browser's last id belongs to the previous run, and the
        # page drops duplicates itself (keyed by `boot` + id).
        bus = self.srv.bus
        q = bus.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # The token rides along so a page left open across a restart can
            # keep posting; only a same-origin page can read this stream.
            self._sse({"kind": "hello", "boot": self.srv.boot, "token": self.srv.token,
                       **self.srv.state()})
            for rec in bus.history():
                self._sse(rec)
            if bus.pose is not None:
                self._sse({"kind": "pose", **bus.pose})
            while not self.srv._stop.is_set():
                # A short keepalive is how a closed window is noticed: the
                # write fails. `bus.clients` decides whether to notify and
                # whether to reopen the window, so it must not lag for long.
                try:
                    msg = q.get(timeout=2)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self._sse(msg)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            bus.unsubscribe(q)
            self.close_connection = True

    def _sse(self, msg: Dict[str, Any]):
        data = json.dumps(msg, default=str, separators=(",", ":"))
        head = f"id: {msg['id']}\n" if "id" in msg and msg.get("kind") != "pose" else ""
        self.wfile.write(f"{head}data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()
