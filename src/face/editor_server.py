"""
src/face/editor_server.py

A tiny local web server so the face editor can read and write real animation
files instead of only downloading JSON.

    python -m src.face edit            # opens the editor in your browser

The editor is still a single self-contained HTML file and still works opened
with a double click (download / import only). This server is what lets it
list every animation (builtin and saved), save into `animations/` where the
robot picks the files up, delete a saved file to fall back to the builtin, and
ask the *Python* interpolator for the pose at a time so the JS preview can be
checked against what the robot will actually draw.

Safety, because this writes files on a request from a web page:
  * binds to 127.0.0.1 only
  * the Host header must be a loopback name, and a browser Origin header must
    match it, so a page on another site (or a DNS-rebinding trick) can't write
  * names are a strict slug; the path is built from the slug, never from input
  * bodies are size-capped and validated before anything touches the disk
  * writes are atomic (temp file + replace)
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .library import BUILTINS, load_library
from .model import (
    Animation, EASINGS, EYE_FIELDS, FACE_FIELDS, SCHEMA_VERSION,
)

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
MAX_BODY = 256 * 1024
MAX_KEYFRAMES = 500
MAX_SAMPLES = 400
LOOPBACK = {"127.0.0.1", "localhost", "[::1]"}


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_animation(d: Any) -> List[str]:
    """Everything wrong with an animation dict, or [] if it is good to save.

    Stricter than `Animation.from_dict` on purpose: the loader tolerates a lot
    (unknown keys, bad types that happen to coerce) so that old files keep
    loading, but a file the editor *writes* should be clean.
    """
    errs: List[str] = []
    if not isinstance(d, dict):
        return ["animation must be a JSON object"]
    name = d.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        errs.append("name must be lowercase letters, digits, '_' or '-' (max 48)")
    if int(d.get("schema", SCHEMA_VERSION) or 0) > SCHEMA_VERSION:
        errs.append(f"schema newer than v{SCHEMA_VERSION}")
    dur = d.get("duration")
    if dur is not None and (not _num(dur) or dur <= 0 or dur > 600):
        errs.append("duration must be a number in (0, 600]")
    pr = d.get("priority", 50)
    if not _num(pr) or not 0 <= pr <= 1000:
        errs.append("priority must be a number in [0, 1000]")
    for flag in ("loop", "interruptible", "blink_suppress", "hold_last"):
        if flag in d and not isinstance(d[flag], bool):
            errs.append(f"{flag} must be true or false")
    tags = d.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) and len(t) <= 32 for t in tags):
        errs.append("tags must be a list of short strings")
    kfs = d.get("keyframes")
    if not isinstance(kfs, list) or not kfs:
        errs.append("keyframes must be a non-empty list")
        return errs
    if len(kfs) > MAX_KEYFRAMES:
        errs.append(f"too many keyframes (max {MAX_KEYFRAMES})")
        return errs
    for i, k in enumerate(kfs):
        if not isinstance(k, dict):
            errs.append(f"keyframe {i}: must be an object")
            continue
        if not _num(k.get("t")) or k["t"] < 0:
            errs.append(f"keyframe {i}: t must be a number >= 0")
        if k.get("ease", "in_out") not in EASINGS:
            errs.append(f"keyframe {i}: unknown ease '{k.get('ease')}'")
        for part, allowed in (("both", EYE_FIELDS), ("left", EYE_FIELDS),
                              ("right", EYE_FIELDS), ("face", FACE_FIELDS)):
            sec = k.get(part, {})
            if not isinstance(sec, dict):
                errs.append(f"keyframe {i}: {part} must be an object")
                continue
            for key, val in sec.items():
                if key not in allowed:
                    errs.append(f"keyframe {i}: unknown {part} parameter '{key}'")
                elif not _num(val):
                    errs.append(f"keyframe {i}: {part}.{key} must be a finite number")
    last = max((k["t"] for k in kfs if isinstance(k, dict) and _num(k.get("t"))), default=0)
    if _num(dur) and dur < last - 1e-6:
        errs.append("duration is shorter than the last keyframe")
    return errs


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------
class Store:
    """The animations directory, as the editor sees it."""

    def __init__(self, directory: str):
        self.dir = os.path.abspath(directory)

    def _files(self) -> Dict[str, str]:
        """name -> path of the file that currently defines it (loader order)."""
        out: Dict[str, str] = {}
        if not os.path.isdir(self.dir):
            return out
        for fn in sorted(os.listdir(self.dir)):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(self.dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    out[Animation.from_dict(json.load(fh)).name] = path
            except Exception:
                continue            # a bad file is the loader's problem, not ours
        return out

    def listing(self) -> List[Dict[str, Any]]:
        files = self._files()
        rows: List[Dict[str, Any]] = []
        for name in sorted(set(BUILTINS) | set(files)):
            a = (self._read(files[name]) if name in files else BUILTINS[name])
            if name in files and name in BUILTINS:
                src = "override"
            elif name in files:
                src = "user"
            else:
                src = "builtin"
            rows.append({"name": name, "source": src, "duration": a.duration,
                         "loop": a.loop, "priority": a.priority, "tags": a.tags})
        return rows

    @staticmethod
    def _read(path: str) -> Animation:
        with open(path, "r", encoding="utf-8") as fh:
            return Animation.from_dict(json.load(fh))

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        files = self._files()
        if name in files:
            return self._read(files[name]).as_dict()
        if name in BUILTINS:
            return BUILTINS[name].as_dict()
        return None

    def save(self, d: Dict[str, Any]) -> str:
        """Write `d` as its own file. Replaces the file that already defines
        this name (even if a person named it differently) so there is never a
        second file fighting it in the loader."""
        name = d["name"]
        os.makedirs(self.dir, exist_ok=True)
        path = self._files().get(name) or os.path.join(self.dir, f"{name}.json")
        # Round-trip through the model so the file is exactly what the robot
        # will read, in the canonical key order.
        text = Animation.from_dict(d).to_json()
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        return path

    def delete(self, name: str) -> bool:
        path = self._files().get(name)
        if not path:
            return False
        os.remove(path)
        return True


# ----------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------
def _host_ok(host_header: str) -> bool:
    host = (host_header or "").strip().lower()
    if host.startswith("["):
        name = host.split("]")[0] + "]"
    else:
        name = host.rsplit(":", 1)[0] if ":" in host else host
    return name in LOOPBACK


def make_server(editor_html: str, animations_dir: str = "animations",
                host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    store = Store(animations_dir)
    html_path = os.path.abspath(editor_html)

    class Handler(BaseHTTPRequestHandler):
        server_version = "ashface"

        def log_message(self, *a):          # quiet
            pass

        # -- helpers ---------------------------------------------------
        def _drain(self):
            """Read and discard an unread request body before replying.

            Replying (and closing) while the client is still sending makes the
            client see a connection reset instead of our error, on some
            platforms every time. Capped, so a huge upload is just dropped.
            """
            if getattr(self, "_body_read", False):
                return
            self._body_read = True
            try:
                left = min(int(self.headers.get("Content-Length", "0")), 4 * MAX_BODY)
            except ValueError:
                return
            while left > 0:
                chunk = self.rfile.read(min(65536, left))
                if not chunk:
                    break
                left -= len(chunk)

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            if code >= 400:
                self._drain()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any):
            self._send(code, json.dumps(obj).encode("utf-8"))

        def _guard(self) -> bool:
            if not _host_ok(self.headers.get("Host", "")):
                self._json(403, {"error": "bad host"})
                return False
            origin = self.headers.get("Origin")
            if origin:
                o = urlparse(origin)
                if (o.hostname or "") not in {"127.0.0.1", "localhost", "::1"} or \
                        (o.netloc or "").lower() != (self.headers.get("Host", "")).lower():
                    self._json(403, {"error": "cross-origin request refused"})
                    return False
            return True

        def _body(self) -> Optional[Any]:
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                self._json(413, {"error": "body too large"})
                return None
            try:
                raw = self.rfile.read(n)
                self._body_read = True
                return json.loads(raw.decode("utf-8"))
            except Exception:
                self._json(400, {"error": "body is not valid JSON"})
                return None

        def _name(self, path: str) -> Optional[str]:
            n = path[len("/api/animation/"):]
            if not NAME_RE.match(n):
                self._json(400, {"error": "bad animation name"})
                return None
            return n

        # -- routes ----------------------------------------------------
        def do_GET(self):
            if not self._guard():
                return
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                try:
                    with open(html_path, "rb") as fh:
                        self._send(200, fh.read(), "text/html; charset=utf-8")
                except OSError:
                    self._json(500, {"error": "editor file missing"})
            elif path == "/api/ping":
                self._json(200, {"ok": True, "dir": store.dir, "schema": SCHEMA_VERSION})
            elif path == "/api/library":
                self._json(200, {"animations": store.listing()})
            elif path.startswith("/api/animation/"):
                name = self._name(path)
                if name is None:
                    return
                d = store.get(name)
                if d is None:
                    self._json(404, {"error": "no such animation"})
                else:
                    self._json(200, d)
            else:
                self._json(404, {"error": "not found"})

        def do_PUT(self):
            if not self._guard():
                return
            path = urlparse(self.path).path
            if not path.startswith("/api/animation/"):
                return self._json(404, {"error": "not found"})
            name = self._name(path)
            if name is None:
                return
            d = self._body()
            if d is None:
                return
            errs = validate_animation(d)
            if not errs and d["name"] != name:
                errs.append("name in the body must match the URL")
            if errs:
                return self._json(422, {"errors": errs})
            saved = store.save(d)
            self._json(200, {"saved": os.path.basename(saved), "name": name})

        def do_DELETE(self):
            if not self._guard():
                return
            path = urlparse(self.path).path
            if not path.startswith("/api/animation/"):
                return self._json(404, {"error": "not found"})
            name = self._name(path)
            if name is None:
                return
            if store.delete(name):
                self._json(200, {"deleted": name, "reverts_to_builtin": name in BUILTINS})
            else:
                self._json(404, {"error": "no saved file for that name"})

        def do_POST(self):
            """POST /api/sample {animation, times} -> the Python interpolator's
            poses, so the editor can verify its JS copy against the robot's."""
            if not self._guard():
                return
            if urlparse(self.path).path != "/api/sample":
                return self._json(404, {"error": "not found"})
            body = self._body()
            if body is None:
                return
            anim_d = body.get("animation") if isinstance(body, dict) else None
            times = body.get("times") if isinstance(body, dict) else None
            errs = validate_animation(anim_d)
            if errs:
                return self._json(422, {"errors": errs})
            if (not isinstance(times, list) or len(times) > MAX_SAMPLES
                    or not all(_num(t) for t in times)):
                return self._json(422, {"errors": ["times must be a list of numbers"]})
            a = Animation.from_dict(anim_d)
            self._json(200, {"poses": [a.sample(float(t)).as_dict() for t in times]})

    return ThreadingHTTPServer((host, port), Handler)


def serve(editor_html: str, animations_dir: str = "animations", port: int = 8765,
          open_browser: bool = True) -> None:
    srv = make_server(editor_html, animations_dir, port=port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Face editor: {url}")
    print(f"Saving to:   {os.path.abspath(animations_dir)}")
    print("Ctrl-C to stop. Restart ASH (or the face) to load saved animations.")
    if open_browser:
        import webbrowser
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
