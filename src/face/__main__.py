"""CLI for the face subsystem. See src/face/__init__.py for usage."""

import sys
import time

from . import BUILTINS, FaceRenderer, build_face, export_builtins, load_library
from .drivers import GifRecorder


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "preview"
    arg = sys.argv[2] if len(sys.argv) > 2 else None

    if cmd == "list":
        lib = load_library()
        for name, a in sorted(lib.items()):
            tags = ",".join(a.tags) or "-"
            print(f"  {name:<16} {a.duration:>5.2f}s  p{a.priority:<3} "
                  f"{'loop' if a.loop else '    '}  {tags}")
        return 0

    if cmd == "export":
        n = export_builtins(arg or "animations/builtin")
        print(f"Exported {n} animations to {arg or 'animations/builtin'}/")
        print("Import any of them in face_editor.html.")
        return 0

    if cmd == "gif":
        name = arg or "happy"
        lib = load_library()
        anim = lib.get(name)
        if anim is None:
            print(f"No animation '{name}'. Try: python -m src.face list")
            return 1
        rec = GifRecorder(path=f"{name}.gif", fps=25, scale=3)
        player = build_face(driver="null")
        player.driver = rec
        player.play(name, force=True)
        dur = (anim.duration or 1.0) * (2 if anim.loop else 1) + 0.4
        t0 = time.time()
        while time.time() - t0 < dur:
            player.render_frame()
            time.sleep(1 / 25)
        print(rec.save())
        return 0

    # preview
    name = arg or "idle"
    player = build_face(driver="terminal", fps=20)
    if name not in player.library:
        print(f"No animation '{name}'. Try: python -m src.face list")
        return 1
    print(f"Playing '{name}'. Ctrl-C to stop.\n")
    player.play(name, force=True)
    try:
        while True:
            player.render_frame()
            time.sleep(1 / 20)
    except KeyboardInterrupt:
        player.shutdown(play_outro=False)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
