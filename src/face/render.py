"""
src/face/render.py

Turns a FaceParams pose into pixels.

One renderer, several outputs: an RGB PIL image for desktop preview, a 1-bit
buffer for monochrome OLEDs, and an ANSI string for the terminal. All three
go through the same geometry so what you see in the editor is what appears on
the panel.

Rendering order per eye:
    1. rounded rect, per-corner radii, at (home + offset + gaze)
    2. vertical scale by `open` about the eye's centre  (blink)
    3. slice off the top by `lid_top`, angled by `lid_angle`  (brow)
    4. slice off the bottom by `lid_bottom`  (squint / smile)

Lids are drawn as background-coloured polygons over the eye rather than by
clipping the eye's path. That is deliberate: an angled lid over a rounded
rectangle produces a shape with no closed-form outline, and covering is both
exact and about ten times less code than trying to boolean it.

Supersampling
-------------
Small OLED faces live or die on edge quality. Everything renders at 3x and
downsamples, which costs microseconds at 128x64 and is the difference between
crisp curves and staircase artefacts. On a 1-bit target the downsample is
thresholded back to pure black and white, which also gives free antialiasing-
by-dithering on the diagonal lid edges.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from .model import FaceParams, REF_HEIGHT, REF_WIDTH

SS = 3   # supersample factor


def _rotate(px: float, py: float, cx: float, cy: float, deg: float) -> Tuple[float, float]:
    if not deg:
        return px, py
    r = math.radians(deg)
    dx, dy = px - cx, py - cy
    return cx + dx * math.cos(r) - dy * math.sin(r), cy + dx * math.sin(r) + dy * math.cos(r)


class FaceRenderer:
    """Renders poses at an arbitrary output size."""

    def __init__(self, width: int = REF_WIDTH, height: int = REF_HEIGHT,
                 fg: Tuple[int, int, int] = (0, 200, 255),
                 bg: Tuple[int, int, int] = (0, 0, 0),
                 supersample: int = SS):
        self.width, self.height = width, height
        self.fg, self.bg = fg, bg
        self.ss = max(1, supersample)
        # Author-space -> output-space scale. Animations are written against
        # 128x64 and stretch to whatever panel is attached.
        self.sx = width / REF_WIDTH
        self.sy = height / REF_HEIGHT

    # ------------------------------------------------------------------
    def render(self, pose: FaceParams):
        """-> PIL.Image in RGB at the configured size."""
        from PIL import Image, ImageDraw

        w, h = self.width * self.ss, self.height * self.ss
        img = Image.new("RGB", (w, h), self.bg)
        draw = ImageDraw.Draw(img)

        b = max(0.0, min(1.0, pose.brightness))
        fg = tuple(int(c * b) for c in self.fg)

        cx, cy = w / 2, h / 2
        gap = pose.gap * self.sx * self.ss * pose.scale

        for side, sign in (("left", -1), ("right", +1)):
            eye = getattr(pose, side)
            if eye.open <= 0.001 or eye.w <= 0 or eye.h <= 0:
                self._draw_closed(draw, pose, eye, cx, cy, gap, sign, fg)
                continue
            self._draw_eye(draw, pose, eye, cx, cy, gap, sign, fg)

        if self.ss > 1:
            img = img.resize((self.width, self.height), Image.LANCZOS)
        if pose.tilt:
            img = img.rotate(pose.tilt, resample=Image.BICUBIC, fillcolor=self.bg)
        return img

    # ------------------------------------------------------------------
    def _geometry(self, pose: FaceParams, eye, cx, cy, gap, sign):
        s = self.ss * pose.scale
        ew = eye.w * self.sx * s
        eh = eye.h * self.sy * s * max(0.0, eye.open)

        ox = (eye.x + pose.look_x) * self.sx * self.ss
        oy = (eye.y + pose.look_y) * self.sy * self.ss

        ecx = cx + sign * (gap / 2 + ew / 2) + ox
        ecy = cy + oy
        return ecx, ecy, ew, eh

    def _draw_eye(self, draw, pose: FaceParams, eye, cx, cy, gap, sign, fg):
        ecx, ecy, ew, eh = self._geometry(pose, eye, cx, cy, gap, sign)
        x0, y0 = ecx - ew / 2, ecy - eh / 2
        x1, y1 = ecx + ew / 2, ecy + eh / 2

        tl, tr, br, bl = eye.corners()
        scale = self.ss * pose.scale * min(self.sx, self.sy)
        # A radius larger than half the shorter side is geometrically
        # impossible; clamp rather than let PIL raise on a slider overshoot.
        rmax = min(ew, eh) / 2
        radii = [max(0.0, min(r * scale, rmax)) for r in (tl, tr, br, bl)]

        self._rounded_rect(draw, (x0, y0, x1, y1), radii, fg, eye.rotation, ecx, ecy)

        # --- lids ------------------------------------------------------
        if eye.lid_top > 0.001 or abs(eye.lid_angle) > 0.01:
            self._lid_top(draw, x0, y0, x1, y1, eye, sign, ecx, ecy)
        if eye.lid_bottom > 0.001:
            cut = eh * min(1.0, eye.lid_bottom)
            pts = [(x0 - 2, y1 - cut), (x1 + 2, y1 - cut), (x1 + 2, y1 + 2), (x0 - 2, y1 + 2)]
            pts = [_rotate(px, py, ecx, ecy, eye.rotation) for px, py in pts]
            draw.polygon(pts, fill=self.bg)

    def _lid_top(self, draw, x0, y0, x1, y1, eye, sign, ecx, ecy):
        """Angled top lid.

        `lid_angle` is anatomical: positive lowers the corner nearest the
        nose on both eyes. Which screen-space edge that is depends on the
        side, and this method is the only place in the codebase that knows
        it -- see Keyframe.resolved_right for why that matters.
        """
        eh = y1 - y0
        ew = x1 - x0
        base = eh * min(1.0, max(0.0, eye.lid_top))
        drop = math.tan(math.radians(eye.lid_angle)) * ew / 2

        inner_y = y0 + base + drop      # toward the nose
        outer_y = y0 + base - drop      # toward the temple

        if sign < 0:      # left eye: the inner edge is its right-hand side
            left_y, right_y = outer_y, inner_y
        else:             # right eye: the inner edge is its left-hand side
            left_y, right_y = inner_y, outer_y

        pts = [(x0 - 2, y0 - 2), (x1 + 2, y0 - 2), (x1 + 2, right_y), (x0 - 2, left_y)]
        pts = [_rotate(px, py, ecx, ecy, eye.rotation) for px, py in pts]
        draw.polygon(pts, fill=self.bg)

    def _draw_closed(self, draw, pose, eye, cx, cy, gap, sign, fg):
        """A fully closed eye is a line, not nothing. Rendering it as empty
        makes a blink look like the display dropped a frame."""
        ecx, ecy, ew, _ = self._geometry(pose, eye, cx, cy, gap, sign)
        t = max(1.0, 1.5 * self.ss)
        draw.rounded_rectangle(
            [ecx - ew / 2, ecy - t, ecx + ew / 2, ecy + t],
            radius=t, fill=fg,
        )

    @staticmethod
    def _rounded_rect(draw, box, radii, fill, rotation, cx, cy):
        x0, y0, x1, y1 = box
        tl, tr, br, bl = radii

        if rotation:
            # PIL cannot rotate a primitive, so approximate the outline as a
            # polygon and rotate the points. Corner arcs get 6 segments each,
            # which is indistinguishable from smooth after downsampling.
            pts = []
            for (cxr, cyr, r, a0) in (
                (x0 + tl, y0 + tl, tl, 180), (x1 - tr, y0 + tr, tr, 270),
                (x1 - br, y1 - br, br, 0), (x0 + bl, y1 - bl, bl, 90),
            ):
                if r <= 0.5:
                    pts.append(_rotate(cxr, cyr, cx, cy, rotation))
                    continue
                for i in range(7):
                    a = math.radians(a0 + 90 * i / 6)
                    pts.append(_rotate(cxr + r * math.cos(a), cyr + r * math.sin(a),
                                       cx, cy, rotation))
            draw.polygon(pts, fill=fill)
            return

        if len({round(r, 2) for r in radii}) == 1:
            draw.rounded_rectangle(box, radius=radii[0], fill=fill)
            return
        try:
            draw.rounded_rectangle(box, radius=max(radii), fill=fill,
                                   corners=(True, True, True, True))
        except TypeError:
            draw.rounded_rectangle(box, radius=max(radii), fill=fill)

    # ------------------------------------------------------------------
    # Alternative outputs
    # ------------------------------------------------------------------
    def render_mono(self, pose: FaceParams, threshold: int = 128):
        """1-bit PIL image for SSD1306-class panels."""
        img = self.render(pose).convert("L")
        return img.point(lambda p: 255 if p >= threshold else 0, mode="1")

    def render_ansi(self, pose: FaceParams, cols: int = 64, rows: int = 20) -> str:
        """Terminal preview using half-block characters.

        Not a toy: it is the only preview that works over SSH on a headless
        robot, and it is how you check an animation actually plays on the
        target machine before wiring up a screen.
        """
        img = self.render(pose).convert("L").resize((cols, rows * 2))
        px = img.load()
        out = []
        for r in range(rows):
            line = []
            for c in range(cols):
                top = px[c, r * 2] > 100
                bot = px[c, r * 2 + 1] > 100
                line.append("█" if top and bot else "▀" if top else "▄" if bot else " ")
            out.append("".join(line))
        return "\n".join(out)

    def render_bytes(self, pose: FaceParams) -> bytes:
        """Packed 1bpp page-major buffer, SSD1306 layout. Lets a driver push
        frames without pulling PIL's image pipeline into the hot loop."""
        img = self.render_mono(pose)
        px = img.load()
        w, h = img.size
        buf = bytearray(w * (h // 8))
        for page in range(h // 8):
            for x in range(w):
                byte = 0
                for bit in range(8):
                    if px[x, page * 8 + bit]:
                        byte |= (1 << bit)
                buf[page * w + x] = byte
        return bytes(buf)
