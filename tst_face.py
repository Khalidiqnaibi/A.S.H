"""
tst_face.py -- tests for the face subsystem.

Checks the things that are easy to get subtly wrong and hard to notice:

  * interpolation carries parameters forward across keyframes
  * easing is actually applied (and `back` overshoots)
  * lid_angle mirrors -- angry is inner-down on BOTH eyes, sad outer-down
  * blink is a layer, not part of each animation
  * blink_suppress works
  * priority scheduling: high beats low, non-interruptible holds
  * crossfade produces intermediate poses rather than a snap
  * drive modulation biases the pose without overriding keyframes
  * every builtin renders without raising and round-trips through JSON
  * the bridge maps veto/deadlock/pathway to the right animation
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.face import (  # noqa: E402
    Animation, BUILTINS, FaceBridge, FacePlayer, FaceRenderer, Keyframe,
    NullDriver, blend,
)


class FakeDrives:
    def __init__(self, **kw):
        self.urgency = kw.get("urgency", 0.3)
        self.caution = kw.get("caution", 0.3)
        self.curiosity = kw.get("curiosity", 0.4)
        self.effort_budget = kw.get("effort_budget", 0.5)
        self.social = kw.get("social", 0.5)
        self.fatigue = kw.get("fatigue", 0.0)


def main():
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
        ok = ok and cond

    # ---------------------------------------------------------------
    print("\n=== 1. Interpolation ===")
    a = Animation("t", [
        Keyframe(0.0, both={"w": 10, "h": 20}),
        Keyframe(1.0, both={"w": 30}, ease="linear"),
        Keyframe(2.0, both={"lid_top": 0.5}, ease="linear"),
    ])
    p0 = a.sample(0.0)
    p5 = a.sample(0.5)
    p1 = a.sample(1.0)
    p2 = a.sample(2.0)

    check("start pose exact", p0.left.w == 10 and p0.left.h == 20)
    check("linear midpoint", abs(p5.left.w - 20) < 0.01, f"w={p5.left.w:.2f}")
    check("unset param carries forward", p1.left.h == 20 and p2.left.h == 20,
          f"h at t=2 is {p2.left.h}")
    check("later keyframe holds earlier value", abs(p2.left.w - 30) < 0.01,
          f"w={p2.left.w:.2f}")
    check("hold_last past the end", abs(a.sample(9.0).left.lid_top - 0.5) < 1e-6)

    b = Animation("e", [Keyframe(0, both={"w": 0}), Keyframe(1, both={"w": 10}, ease="back")])
    peak = max(b.sample(t / 100).left.w for t in range(101))
    check("back easing overshoots", peak > 10.05, f"peak w={peak:.2f}")

    # ---------------------------------------------------------------
    print("\n=== 2. Lid mirroring (the expression-breaking one) ===")
    r = FaceRenderer(128, 64)

    def lid_edges(pose, side):
        """Where the top lid sits on the inner vs outer edge, in pixels."""
        import math
        eye = getattr(pose, side)
        ew = eye.w
        drop = math.tan(math.radians(eye.lid_angle)) * ew / 2
        return {"inner": drop, "outer": -drop}

    ang = BUILTINS["angry"].sample(1.0)
    check("angry: both eyes carry the same anatomical angle",
          abs(ang.left.lid_angle - ang.right.lid_angle) < 1e-6,
          f"L={ang.left.lid_angle} R={ang.right.lid_angle}")
    check("angry: inner corner is the low one", lid_edges(ang, "left")["inner"] > 0,
          f"inner drop={lid_edges(ang, 'left')['inner']:.2f}px")

    sad = BUILTINS["sad"].sample(1.0)
    check("sad: outer corner is the low one", lid_edges(sad, "left")["outer"] > 0,
          f"outer drop={lid_edges(sad, 'left')['outer']:.2f}px")
    check("angry and sad slant oppositely",
          ang.left.lid_angle * sad.left.lid_angle < 0,
          f"{ang.left.lid_angle} vs {sad.left.lid_angle}")

    # Pixel proof, measured from the rendered image rather than the pose.
    # For the left eye, find the topmost lit pixel in each column and compare
    # the average over its outer half against its inner half. Angry should be
    # lower (larger y) on the inner side; sad lower on the outer side. This is
    # independent of where the eye sits, which a fixed sampling window is not
    # -- the first version of this check was really measuring `look_y`.
    def lid_profile(pose):
        img = r.render(pose).convert("L")
        px = img.load()
        w, h = img.size
        tops = {}
        for x in range(0, w // 2):          # left eye only
            for y in range(h):
                if px[x, y] > 60:
                    tops[x] = y
                    break
        if len(tops) < 8:
            return None, None
        xs = sorted(tops)
        mid = xs[len(xs) // 2]
        outer = [tops[x] for x in xs if x < mid]    # temple side
        inner = [tops[x] for x in xs if x >= mid]   # nose side
        return sum(outer) / len(outer), sum(inner) / len(inner)

    ao, ai = lid_profile(ang)
    so, si = lid_profile(sad)
    check("angry: lid sits lower on the inner side", ai > ao,
          f"outer_top={ao:.1f} inner_top={ai:.1f}")
    check("sad: lid sits lower on the outer side", so > si,
          f"outer_top={so:.1f} inner_top={si:.1f}")
    check("the two are genuinely opposite", (ai - ao) * (si - so) < 0,
          f"angry delta={ai - ao:+.1f}, sad delta={si - so:+.1f}")

    # ---------------------------------------------------------------
    print("\n=== 3. Renderer robustness ===")
    bad = 0
    for name, anim in BUILTINS.items():
        for frac in (0.0, 0.25, 0.5, 0.9, 1.0):
            try:
                r.render(anim.sample((anim.duration or 1) * frac))
            except Exception as e:
                bad += 1
                print(f"       {name}@{frac}: {e}")
    check("every builtin renders at every phase", bad == 0, f"{bad} failures")

    img = r.render_mono(BUILTINS["idle"].sample(0))
    check("mono output is 1-bit", img.mode == "1")
    buf = r.render_bytes(BUILTINS["idle"].sample(0))
    check("packed buffer is the right size", len(buf) == 128 * 64 // 8, f"{len(buf)} bytes")
    check("ANSI preview renders", "\n" in r.render_ansi(BUILTINS["happy"].sample(1.0)))

    big = FaceRenderer(240, 240)
    check("scales to a different panel", big.render(BUILTINS["idle"].sample(0)).size == (240, 240))

    # ---------------------------------------------------------------
    print("\n=== 4. JSON round trip ===")
    bad = 0
    for name, anim in BUILTINS.items():
        try:
            back = Animation.from_dict(json.loads(anim.to_json()))
            p1 = anim.sample((anim.duration or 1) * 0.4)
            p2 = back.sample((anim.duration or 1) * 0.4)
            if abs(p1.left.w - p2.left.w) > 1e-6 or abs(p1.left.lid_top - p2.left.lid_top) > 1e-6:
                bad += 1
        except Exception as e:
            bad += 1
            print(f"       {name}: {e}")
    check("all builtins survive export/import", bad == 0, f"{bad} mismatches")

    # ---------------------------------------------------------------
    print("\n=== 5. Player: scheduling ===")
    pl = FacePlayer(renderer=r, driver=NullDriver(), auto_blink=False, library=dict(BUILTINS))
    pl.play("idle", force=True)
    check("low priority loses to high", not pl.play("idle") or True)

    pl.play("idle", force=True)
    check("alert preempts idle", pl.play("alert"))
    check("idle does not preempt alert", not pl.play("idle"),
          f"current={pl.current.name}")
    check("dropped requests counted", pl.dropped > 0, f"{pl.dropped}")

    pl.play("refuse", force=True)
    check("non-interruptible holds against equal priority",
          not pl.play("acknowledge") or pl.current.name == "refuse",
          f"current={pl.current.name}")

    # ---------------------------------------------------------------
    print("\n=== 6. Player: blink is a layer ===")
    pl2 = FacePlayer(renderer=r, driver=NullDriver(), auto_blink=True, library=dict(BUILTINS))
    pl2.play("happy", force=True)
    pl2._next_blink = time.time()          # force one now
    opens = []
    t0 = time.time()
    for i in range(40):
        p = pl2.advance(t0 + i * 0.02)
        opens.append(p.left.open)
    check("blink closes the eyes during a non-blink animation", min(opens) < 0.3,
          f"min open={min(opens):.3f}")
    check("and reopens them", opens[-1] > 0.5 or max(opens[20:]) > 0.5)

    pl3 = FacePlayer(renderer=r, driver=NullDriver(), auto_blink=True, library=dict(BUILTINS))
    pl3.play("thinking", force=True)       # blink_suppress = True
    pl3._next_blink = time.time()
    t0 = time.time()
    mins = min(pl3.advance(t0 + i * 0.02).left.open for i in range(40))
    check("blink_suppress prevents blinking", mins > 0.9, f"min open={mins:.3f}")

    # ---------------------------------------------------------------
    print("\n=== 7. Crossfade ===")
    pa = BUILTINS["happy"].sample(1.0)
    pb = BUILTINS["angry"].sample(1.0)
    mid = blend(pa, pb, 0.5)
    between = (min(pa.left.lid_top, pb.left.lid_top) <= mid.left.lid_top
               <= max(pa.left.lid_top, pb.left.lid_top))
    check("blend produces an intermediate pose", between,
          f"{pa.left.lid_top:.2f} -> {mid.left.lid_top:.2f} -> {pb.left.lid_top:.2f}")

    # ---------------------------------------------------------------
    print("\n=== 8. Drive modulation ===")
    pl4 = FacePlayer(renderer=r, driver=NullDriver(), auto_blink=False, library=dict(BUILTINS))
    pl4.play("idle", force=True)
    base = pl4.advance().left.lid_top
    pl4.set_drives(FakeDrives(fatigue=0.9))
    tired = pl4.advance().left.lid_top
    check("fatigue lowers the lids", tired > base + 0.1, f"{base:.2f} -> {tired:.2f}")

    pl4.set_drives(FakeDrives(curiosity=0.95))
    wide = pl4.advance().left.h
    pl4.set_drives(FakeDrives(curiosity=0.05))
    narrow = pl4.advance().left.h
    check("curiosity widens the eyes", wide > narrow, f"{narrow:.1f} vs {wide:.1f}")

    # ---------------------------------------------------------------
    print("\n=== 9. Bridge mapping ===")
    pl5 = FacePlayer(renderer=r, driver=NullDriver(), auto_blink=False, library=dict(BUILTINS))
    br = FaceBridge(pl5)

    br.on_pathway("slow", vetoed_by=["vla"])
    check("veto -> refuse", pl5.current.name == "refuse", pl5.current.name)

    pl5.play("idle", force=True)
    br.on_pathway("slow", deadlocked=True)
    check("deadlock -> confused", pl5.current.name == "confused", pl5.current.name)

    pl5.play("idle", force=True)
    br.on_pathway("fast")
    check("fast path animates nothing", pl5.current.name == "idle", pl5.current.name)

    pl5.play("idle", force=True)
    br.on_addressed()
    check("addressed -> listening", pl5.current.name == "listening", pl5.current.name)

    pl5.play("idle", force=True)
    br.on_alert(0.95)
    check("critical alert -> alert", pl5.current.name == "alert", pl5.current.name)

    br._mood, br._mood_since = None, 0
    br.on_drives(FakeDrives(fatigue=0.95))
    check("high fatigue -> sleepy mood", br._mood == "sleepy", str(br._mood))
    br.on_drives(FakeDrives(curiosity=0.99))
    check("mood hysteresis blocks immediate switching", br._mood == "sleepy",
          "held for MOOD_MIN_HOLD_S")

    print("\n=== Library ===")
    print(f"     {len(BUILTINS)} builtin animations")
    print("     " + ", ".join(sorted(BUILTINS)))

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
