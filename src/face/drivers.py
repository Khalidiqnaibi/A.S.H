"""
src/face/drivers.py

Where the face actually appears.

Every driver takes a PIL image and puts it somewhere. They all degrade the
same way as the sensors: probe at construction, report honestly, never crash
the player. `pick_driver()` chooses the best available, so the same code runs
on a Pi with an OLED soldered to it and on a laptop with nothing attached.

Ordering in `pick_driver` is by fidelity, not convenience: a real panel beats
a window, a window beats the terminal, and the terminal always works. The
terminal one is not a joke -- it is how you verify an animation on a headless
robot over SSH before wiring a screen.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from typing import Any, Optional

logger = logging.getLogger("ash.face.drivers")


class BaseDriver:
    name = "base"
    available = False

    def show(self, img, pose=None):
        raise NotImplementedError

    def close(self):
        pass


class NullDriver(BaseDriver):
    """Renders nothing. Used in tests and when the face is disabled."""

    name = "null"
    available = True

    def show(self, img, pose=None):
        pass


class TerminalDriver(BaseDriver):
    """ANSI half-block preview. Works over SSH, works everywhere."""

    name = "terminal"

    def __init__(self, cols: int = 64, rows: int = 18, throttle_fps: int = 12):
        self.cols, self.rows = cols, rows
        self.available = sys.stdout.isatty()
        self._min_dt = 1.0 / max(1, throttle_fps)
        self._last = 0.0
        self._primed = False

    def show(self, img, pose=None):
        now = time.time()
        # A terminal cannot keep up with 30fps of full redraws and the
        # flicker is worse than the lower rate.
        if now - self._last < self._min_dt:
            return
        self._last = now

        g = img.convert("L").resize((self.cols, self.rows * 2))
        px = g.load()
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                t = px[c, r * 2] > 100
                b = px[c, r * 2 + 1] > 100
                row.append("█" if t and b else "▀" if t else "▄" if b else " ")
            lines.append("".join(row))

        if self._primed:
            sys.stdout.write(f"\033[{self.rows}A")   # cursor up, no clear = no flicker
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self._primed = True

    def close(self):
        if self._primed:
            sys.stdout.write("\033[0m\n")
            sys.stdout.flush()


class WindowDriver(BaseDriver):
    """Desktop preview window via Tkinter. No extra dependency on most
    installs, and unlike pygame it does not want to own the event loop."""

    name = "window"

    def __init__(self, scale: int = 4, title: str = "A.S.H"):
        self.scale = scale
        self.available = False
        self._root = None
        try:
            import tkinter as tk
            from PIL import ImageTk  # noqa: F401

            self._tk = tk
            self._root = tk.Tk()
            self._root.title(title)
            self._root.configure(bg="black")
            self._label = tk.Label(self._root, bd=0, bg="black")
            self._label.pack()
            self.available = True
        except Exception as e:
            logger.info("Window driver unavailable: %s", e)

    def show(self, img, pose=None):
        if not self.available:
            return
        from PIL import Image, ImageTk

        w, h = img.size
        big = img.resize((w * self.scale, h * self.scale), Image.NEAREST)
        photo = ImageTk.PhotoImage(big)
        self._label.configure(image=photo)
        self._label.image = photo    # keep a reference or Tk garbage-collects it
        self._root.update_idletasks()
        self._root.update()

    def close(self):
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass


class OLEDDriver(BaseDriver):
    """SSD1306/SSD1309 over I2C. The classic 128x64 robot-eye panel."""

    name = "oled"

    def __init__(self, width: int = 128, height: int = 64, address: int = 0x3C,
                 bus: int = 1):
        self.width, self.height = width, height
        self.available = False
        self._dev = None
        try:
            from board import SCL, SDA  # type: ignore
            import busio  # type: ignore
            import adafruit_ssd1306  # type: ignore

            i2c = busio.I2C(SCL, SDA)
            self._dev = adafruit_ssd1306.SSD1306_I2C(width, height, i2c, addr=address)
            self._dev.fill(0)
            self._dev.show()
            self.available = True
            logger.info("OLED %dx%d at 0x%02X", width, height, address)
        except Exception as e:
            logger.info("OLED unavailable: %s", e)

    def show(self, img, pose=None):
        if not self.available:
            return
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height))
        self._dev.image(img.convert("1"))
        self._dev.show()

    def close(self):
        if self.available:
            try:
                self._dev.fill(0)
                self._dev.show()
            except Exception:
                pass


class TFTDriver(BaseDriver):
    """Colour SPI panel (ST7789 / ILI9341) via luma.lcd or Adafruit RGB
    display. Colour is worth having: brightness modulation and the alert
    flash read far better than on mono."""

    name = "tft"

    def __init__(self, width: int = 240, height: int = 240, rotation: int = 0):
        self.width, self.height = width, height
        self.available = False
        self._dev = None
        try:
            import board  # type: ignore
            import digitalio  # type: ignore
            from adafruit_rgb_display import st7789  # type: ignore

            spi = board.SPI()
            self._dev = st7789.ST7789(
                spi,
                cs=digitalio.DigitalInOut(board.CE0),
                dc=digitalio.DigitalInOut(board.D25),
                rst=None, baudrate=64_000_000,
                width=width, height=height, rotation=rotation,
            )
            self.available = True
            logger.info("TFT %dx%d ready", width, height)
        except Exception as e:
            logger.info("TFT unavailable: %s", e)

    def show(self, img, pose=None):
        if not self.available:
            return
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height))
        self._dev.image(img)

    def close(self):
        pass


class GifRecorder(BaseDriver):
    """Records frames to an animated GIF. Not a display -- a debugging and
    documentation tool. Being able to hand someone a GIF of exactly what the
    robot's face did is worth the twenty lines."""

    name = "gif"
    available = True

    def __init__(self, path: str = "face.gif", fps: int = 25, max_frames: int = 600,
                 scale: int = 3):
        self.path, self.fps, self.max_frames, self.scale = path, fps, max_frames, scale
        self.frames: list = []

    def show(self, img, pose=None):
        if len(self.frames) >= self.max_frames:
            return
        from PIL import Image
        w, h = img.size
        self.frames.append(img.resize((w * self.scale, h * self.scale), Image.NEAREST))

    def save(self) -> Optional[str]:
        if not self.frames:
            return None
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.frames[0].save(
            self.path, save_all=True, append_images=self.frames[1:],
            duration=int(1000 / self.fps), loop=0, optimize=True,
        )
        logger.info("Wrote %s (%d frames)", self.path, len(self.frames))
        return self.path

    def close(self):
        self.save()


def pick_driver(prefer: Optional[str] = None, **kw) -> BaseDriver:
    """Best available output. `prefer` forces a specific one by name."""
    candidates = {
        "oled": OLEDDriver, "tft": TFTDriver, "window": WindowDriver,
        "terminal": TerminalDriver, "gif": GifRecorder, "null": NullDriver,
    }

    if prefer:
        cls = candidates.get(prefer)
        if cls is None:
            logger.warning("Unknown driver %r; falling back", prefer)
        else:
            d = cls(**kw) if prefer != "null" else NullDriver()
            if d.available:
                return d
            logger.warning("Requested driver %r unavailable; falling back", prefer)

    for name in ("oled", "tft", "window", "terminal"):
        try:
            d = candidates[name]()
            if d.available:
                logger.info("Face driver: %s", name)
                return d
        except Exception:
            continue
    logger.info("Face driver: null (nothing to draw on)")
    return NullDriver()
