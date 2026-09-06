"""
src/face/model.py

The face parameter model. This file is the contract.

Everything that renders a face -- the Python renderer, the HTML editor's
canvas preview, any future firmware port -- must agree on exactly these
parameters and exactly this interpolation. If they diverge, animations look
right in the editor and wrong on the robot, which is the single most annoying
failure mode a tool like this can have. So the schema lives here, in one
place, and `SCHEMA_VERSION` is written into every exported file.

Design: expression from geometry, not sprites
---------------------------------------------
There is no mouth, no eyebrows, no bitmap assets. Everything is two rounded
rectangles, and every emotion is produced by four levers:

    SHAPE     width, height, per-corner radius
    POSITION  offset from home, plus a whole-face look vector
    LIDS      top/bottom coverage as a fraction of eye height
    SLANT     lid angle in degrees

Slant is what earns its keep. An angled top lid reads unmistakably as an
eyebrow -- inward-down is angry, outward-down is sad -- without any separate
brow object to position, animate, and keep in sync. One number replaces an
entire second layer of art.

`lid_angle` is anatomical, not geometric: a single positive value means
"inner corners down" on *both* eyes, and the renderer works out that this is
opposite screen-space directions for left and right. Authors never mirror
anything by hand, which is the only way symmetric expressions stay symmetric
when someone edits one keyframe six months later.

Interpolation
-------------
Keyframes hold *partial* parameter sets. A keyframe that only specifies
`lid_top` leaves everything else to be carried forward. This keeps hand-
written animations short and readable, and means adding a new parameter later
doesn't invalidate existing files.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

# Canonical display size. Everything is authored against this and scaled at
# render time, so one animation file drives a 128x64 OLED and a 320x240 TFT.
REF_WIDTH = 128
REF_HEIGHT = 64


# ----------------------------------------------------------------------
# Easing
# ----------------------------------------------------------------------
def _clamp01(t: float) -> float:
    return 0.0 if t < 0 else (1.0 if t > 1 else t)


EASINGS: Dict[str, Any] = {
    "linear": lambda t: t,
    "in": lambda t: t * t,
    "out": lambda t: 1 - (1 - t) ** 2,
    "in_out": lambda t: 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2,
    "in_cubic": lambda t: t ** 3,
    "out_cubic": lambda t: 1 - (1 - t) ** 3,
    # Overshoot. Used for surprise and for eyes snapping to a target -- real
    # saccades overshoot slightly and settle, and copying that is most of what
    # makes synthetic eye movement stop looking like a slider.
    "back": lambda t: 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2,
    "elastic": lambda t: (0.0 if t == 0 else 1.0 if t == 1 else
                          -(2 ** (10 * t - 10)) * math.sin((t * 10 - 10.75) * 2.0944)),
    "bounce": lambda t: _bounce(t),
    # Instant switch at the end of the segment: for hard cuts.
    "step": lambda t: 0.0 if t < 1.0 else 1.0,
}


def _bounce(t: float) -> float:
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    if t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


def ease(name: str, t: float) -> float:
    return EASINGS.get(name or "linear", EASINGS["linear"])(_clamp01(t))


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
@dataclass
class EyeParams:
    """One eye. All distances in reference pixels, angles in degrees."""

    x: float = 0.0            # offset from home position
    y: float = 0.0
    w: float = 36.0
    h: float = 36.0
    radius: float = 10.0      # uniform corner radius
    radius_tl: Optional[float] = None   # per-corner overrides
    radius_tr: Optional[float] = None
    radius_br: Optional[float] = None
    radius_bl: Optional[float] = None
    lid_top: float = 0.0      # 0..1 fraction of height hidden from the top
    lid_bottom: float = 0.0
    # Top lid slant in degrees, interpreted anatomically rather than
    # geometrically: POSITIVE lowers the INNER corner (toward the nose) on
    # both eyes, which reads as angry. NEGATIVE lowers the outer corners,
    # which reads as sad. One value, correctly mirrored by the renderer.
    lid_angle: float = 0.0
    open: float = 1.0         # vertical scale; 0 = fully closed (blink)
    rotation: float = 0.0     # whole-eye rotation, deg

    def corners(self) -> Tuple[float, float, float, float]:
        r = self.radius
        return (
            self.radius_tl if self.radius_tl is not None else r,
            self.radius_tr if self.radius_tr is not None else r,
            self.radius_br if self.radius_br is not None else r,
            self.radius_bl if self.radius_bl is not None else r,
        )


@dataclass
class FaceParams:
    """A complete face pose: two eyes plus global transform."""

    left: EyeParams = field(default_factory=EyeParams)
    right: EyeParams = field(default_factory=EyeParams)
    look_x: float = 0.0       # whole-face gaze offset, applied to both eyes
    look_y: float = 0.0
    gap: float = 20.0         # space between the eyes
    scale: float = 1.0
    brightness: float = 1.0   # 0..1, drives colour intensity / OLED contrast
    tilt: float = 0.0         # whole-face rotation, deg

    def copy(self) -> "FaceParams":
        return FaceParams(
            left=replace(self.left), right=replace(self.right),
            look_x=self.look_x, look_y=self.look_y, gap=self.gap,
            scale=self.scale, brightness=self.brightness, tilt=self.tilt,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "left": asdict(self.left), "right": asdict(self.right),
            "look_x": self.look_x, "look_y": self.look_y, "gap": self.gap,
            "scale": self.scale, "brightness": self.brightness, "tilt": self.tilt,
        }


REST = FaceParams()

# Parameters that live on the eye vs. on the face. Used by the interpolator
# and by the editor to build its control panel, so adding a parameter in one
# place makes it appear in both.
EYE_FIELDS = [f for f in EyeParams.__dataclass_fields__]
FACE_FIELDS = ["look_x", "look_y", "gap", "scale", "brightness", "tilt"]


# ----------------------------------------------------------------------
# Keyframes and animations
# ----------------------------------------------------------------------
@dataclass
class Keyframe:
    """A partial pose at a point in time.

    `left` / `right` hold only the eye parameters this keyframe actually sets;
    anything absent is carried forward from the previous keyframe. `both` is
    sugar that writes to each eye -- most animations are symmetric and having
    to specify each eye twice makes files twice as long and twice as easy to
    desynchronise.
    """

    t: float
    both: Dict[str, float] = field(default_factory=dict)
    left: Dict[str, float] = field(default_factory=dict)
    right: Dict[str, float] = field(default_factory=dict)
    face: Dict[str, float] = field(default_factory=dict)
    ease: str = "in_out"

    def resolved_left(self) -> Dict[str, float]:
        d = dict(self.both)
        d.update(self.left)
        return d

    def resolved_right(self) -> Dict[str, float]:
        """No mirroring happens here, deliberately.

        An earlier version negated `lid_angle` for the right eye AND the
        renderer swapped inner/outer by side -- the two cancelled and every
        expression rendered with parallel slants instead of mirrored ones.
        Mirroring now lives in exactly one place: the renderer's `_lid_top`,
        which knows which side of each eye faces the nose. `lid_angle` is
        therefore a single anatomical value, identical for both eyes.
        """
        d = dict(self.both)
        d.update(self.right)
        return d

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"t": self.t, "ease": self.ease}
        for k in ("both", "left", "right", "face"):
            v = getattr(self, k)
            if v:
                out[k] = v
        return out

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Keyframe":
        return Keyframe(
            t=float(d.get("t", 0.0)),
            both=dict(d.get("both", {})),
            left=dict(d.get("left", {})),
            right=dict(d.get("right", {})),
            face=dict(d.get("face", {})),
            ease=d.get("ease", "in_out"),
        )


@dataclass
class Animation:
    """A named sequence of keyframes.

    `priority` decides what preempts what. `interruptible` decides whether an
    equal-priority request can cut it short -- a blink is interruptible, an
    error flash is not, because a half-played error reads as a glitch.

    `blink_suppress` is worth explaining: during concentration people blink
    markedly less, and letting the auto-blinker fire on top of a `thinking`
    animation destroys the impression of focus. Any animation can suppress it.
    """

    name: str
    keyframes: List[Keyframe] = field(default_factory=list)
    duration: Optional[float] = None    # inferred from last keyframe if None
    loop: bool = False
    priority: int = 50
    interruptible: bool = True
    blink_suppress: bool = False
    hold_last: bool = True              # stay on the final pose when done
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.keyframes.sort(key=lambda k: k.t)
        if self.duration is None:
            self.duration = self.keyframes[-1].t if self.keyframes else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "name": self.name,
            "duration": self.duration,
            "loop": self.loop,
            "priority": self.priority,
            "interruptible": self.interruptible,
            "blink_suppress": self.blink_suppress,
            "hold_last": self.hold_last,
            "tags": self.tags,
            "keyframes": [k.as_dict() for k in self.keyframes],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Animation":
        v = int(d.get("schema", SCHEMA_VERSION))
        if v > SCHEMA_VERSION:
            raise ValueError(
                f"animation '{d.get('name')}' uses schema v{v}; this build reads v{SCHEMA_VERSION}"
            )
        return Animation(
            name=d.get("name", "unnamed"),
            keyframes=[Keyframe.from_dict(k) for k in d.get("keyframes", [])],
            duration=d.get("duration"),
            loop=bool(d.get("loop", False)),
            priority=int(d.get("priority", 50)),
            interruptible=bool(d.get("interruptible", True)),
            blink_suppress=bool(d.get("blink_suppress", False)),
            hold_last=bool(d.get("hold_last", True)),
            tags=list(d.get("tags", [])),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)

    # ------------------------------------------------------------------
    def sample(self, t: float, base: Optional[FaceParams] = None) -> FaceParams:
        """Pose at time `t`. This is the interpolator both renderers implement.

        Semantics, precisely, because the editor has to match:
          * before the first keyframe -> the base pose
          * between keyframes -> per-parameter lerp, eased by the *later*
            keyframe's `ease`
          * a parameter absent from the later keyframe holds its accumulated
            value rather than snapping back to base
          * after the last keyframe -> last pose if hold_last, else base
        """
        pose = (base or REST).copy()
        if not self.keyframes:
            return pose

        dur = self.duration or self.keyframes[-1].t or 1e-9
        if self.loop and dur > 0:
            t = t % dur

        # Accumulate every keyframe strictly before t, so carried-forward
        # values are correct even when a parameter was last set five frames ago.
        prev_vals_l: Dict[str, float] = {}
        prev_vals_r: Dict[str, float] = {}
        prev_vals_f: Dict[str, float] = {}
        prev_t = 0.0
        nxt: Optional[Keyframe] = None

        for kf in self.keyframes:
            if kf.t <= t:
                prev_vals_l.update(kf.resolved_left())
                prev_vals_r.update(kf.resolved_right())
                prev_vals_f.update(kf.face)
                prev_t = kf.t
            else:
                nxt = kf
                break

        if nxt is None:
            if self.hold_last:
                _apply(pose, prev_vals_l, prev_vals_r, prev_vals_f)
            return pose

        span = max(1e-9, nxt.t - prev_t)
        raw = (t - prev_t) / span
        u = ease(nxt.ease, raw)

        tgt_l, tgt_r, tgt_f = nxt.resolved_left(), nxt.resolved_right(), nxt.face

        cur_l = dict(prev_vals_l)
        cur_r = dict(prev_vals_r)
        cur_f = dict(prev_vals_f)

        for k, v in tgt_l.items():
            cur_l[k] = _lerp(prev_vals_l.get(k, _default_eye(k)), v, u)
        for k, v in tgt_r.items():
            cur_r[k] = _lerp(prev_vals_r.get(k, _default_eye(k)), v, u)
        for k, v in tgt_f.items():
            cur_f[k] = _lerp(prev_vals_f.get(k, _default_face(k)), v, u)

        _apply(pose, cur_l, cur_r, cur_f)
        return pose

    def finished(self, t: float) -> bool:
        return (not self.loop) and t >= (self.duration or 0.0)


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _default_eye(key: str) -> float:
    v = getattr(EyeParams(), key, 0.0)
    return 0.0 if v is None else float(v)


def _default_face(key: str) -> float:
    return float(getattr(FaceParams(), key, 0.0))


def _apply(pose: FaceParams, left: Dict[str, float],
           right: Dict[str, float], face: Dict[str, float]):
    for k, v in left.items():
        if k in EYE_FIELDS:
            setattr(pose.left, k, v)
    for k, v in right.items():
        if k in EYE_FIELDS:
            setattr(pose.right, k, v)
    for k, v in face.items():
        if k in FACE_FIELDS:
            setattr(pose, k, v)


# ----------------------------------------------------------------------
def blend(a: FaceParams, b: FaceParams, u: float) -> FaceParams:
    """Crossfade two poses. Used when one animation replaces another --
    without it, switching from `happy` to `alert` is a single-frame snap that
    looks like a rendering bug rather than a change of mood."""
    u = _clamp01(u)
    out = FaceParams()
    for side in ("left", "right"):
        ea, eb, eo = getattr(a, side), getattr(b, side), getattr(out, side)
        for k in EYE_FIELDS:
            va, vb = getattr(ea, k), getattr(eb, k)
            if va is None or vb is None:
                setattr(eo, k, vb if u > 0.5 else va)
            else:
                setattr(eo, k, _lerp(float(va), float(vb), u))
    for k in FACE_FIELDS:
        setattr(out, k, _lerp(getattr(a, k), getattr(b, k), u))
    return out
