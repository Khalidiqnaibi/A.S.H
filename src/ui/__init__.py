"""
ASH console: the window where ASH's words, your words and its actions appear.

    bus      thread-safe event stream (chat, cues, face poses, status)
    server   loopback HTTP server: the page, a server-sent event stream, and
             a small JSON API for typing, quick replies and controls
    hooks    wires the bus into a running AmbientRuntime by wrapping, the
             same way the face bridge does, so the ambient loop never knows
             a window exists and a UI failure cannot take it down
    window   opens the console as its own app-style window

The page itself is static/console.html: one file, no build step, themes in CSS.
"""

from .bus import UIBus
from .server import ConsoleServer
from .hooks import attach_ui
from .window import open_window

__all__ = ["UIBus", "ConsoleServer", "attach_ui", "open_window"]
