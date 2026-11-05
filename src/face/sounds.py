"""
src/face/sounds.py

A small sound for some of the face's expressions: a two-note rise for
curious, a bright three-note for very happy, a soft two-note for happy, a
falling pair for alert. Synthesized here (sine notes with a fade, a quarter
volume), written once to a temp file and played asynchronously, so nothing
is shipped and nothing blocks.

Off unless face.sounds is on. Only for lines ASH volunteers -- an answer you
asked for needs no chime -- never in quiet hours, and at most one every 20
seconds.
"""

from __future__ import annotations

import io
import logging
import math
import os
import struct
import tempfile
import time
import wave
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.face.sounds")

RATE = 22050
TONES: Dict[str, List[Tuple[float, float]]] = {      # expression -> [(Hz, seconds)]
    "curious": [(587.0, 0.08), (784.0, 0.12)],
    "happy": [(659.0, 0.08), (880.0, 0.11)],
    "very_happy": [(659.0, 0.07), (880.0, 0.07), (1175.0, 0.12)],
    "alert": [(880.0, 0.09), (622.0, 0.14)],
}
MIN_GAP_S = 20.0


def make_wav(notes: List[Tuple[float, float]], volume: float = 0.25) -> bytes:
    """A mono 16-bit WAV of the notes, each faded in and out."""
    frames = bytearray()
    for hz, secs in notes:
        n = int(RATE * secs)
        fade = max(1, int(RATE * 0.012))
        for i in range(n):
            env = min(1.0, i / fade, (n - 1 - i) / fade)
            frames += struct.pack("<h", int(32767 * volume * env * math.sin(2 * math.pi * hz * i / RATE)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _winsound_play(path: str) -> bool:
    try:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        return True
    except Exception:
        return False


class FaceSounds:
    def __init__(self, enabled: bool = False, play: Callable[[str], bool] = _winsound_play,
                 clock: Callable[[], float] = time.time, folder: Optional[str] = None):
        self.enabled = enabled
        self.play_file = play
        self.clock = clock
        self.folder = folder or os.path.join(tempfile.gettempdir(), "ash-face-sounds")
        self._files: Dict[str, str] = {}
        self._last = -1e18
        self.stats = {"played": 0}

    def _file(self, expr: str) -> Optional[str]:
        if expr in self._files and os.path.exists(self._files[expr]):
            return self._files[expr]
        notes = TONES.get(expr)
        if not notes:
            return None
        try:
            os.makedirs(self.folder, exist_ok=True)
            path = os.path.join(self.folder, f"{expr}.wav")
            with open(path, "wb") as fh:
                fh.write(make_wav(notes))
            self._files[expr] = path
            return path
        except OSError:
            logger.debug("Could not write the %s sound", expr, exc_info=True)
            return None

    def maybe(self, expr: Optional[str], volunteered: bool, quiet: bool) -> bool:
        """Play the sound for `expr` if this is the moment for one."""
        if not self.enabled or not expr or expr not in TONES or not volunteered or quiet:
            return False
        now = self.clock()
        if now - self._last < MIN_GAP_S:
            return False
        path = self._file(expr)
        if path is None or not self.play_file(path):
            return False
        self._last = now
        self.stats["played"] += 1
        return True
