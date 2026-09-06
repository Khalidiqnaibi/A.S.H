"""
ASH robot face: a keyframe animation system for two-eye OLED/TFT faces.

    model     parameter schema, keyframes, easing, interpolation (the contract)
    render    pose -> pixels (RGB, 1-bit, ANSI, packed buffer)
    library   premade animations, keyed to ASH's cognitive states
    player    playback: priority scheduling, layered auto-blink, drive bias
    drivers   output targets (terminal, window, OLED, TFT, GIF)
    bridge    when each animation plays, wired to the brain and the daemon

    face_editor.html   author new animations; exports this module's JSON

CLI:
    python -m src.face preview [name]   play an animation in the terminal
    python -m src.face export           dump builtins to animations/builtin/
    python -m src.face gif <name>       render an animation to a GIF
    python -m src.face list             list available animations
"""

from .model import (
    Animation, EyeParams, FaceParams, Keyframe, REST, SCHEMA_VERSION,
    REF_WIDTH, REF_HEIGHT, blend, ease,
)
from .render import FaceRenderer
from .library import BUILTINS, load_library, export_builtins
from .player import FacePlayer
from .drivers import (
    BaseDriver, NullDriver, TerminalDriver, WindowDriver, OLEDDriver,
    TFTDriver, GifRecorder, pick_driver,
)
from .bridge import FaceBridge, attach_to_daemon

__all__ = [
    "Animation", "EyeParams", "FaceParams", "Keyframe", "REST",
    "SCHEMA_VERSION", "REF_WIDTH", "REF_HEIGHT", "blend", "ease",
    "FaceRenderer", "BUILTINS", "load_library", "export_builtins",
    "FacePlayer", "BaseDriver", "NullDriver", "TerminalDriver", "WindowDriver",
    "OLEDDriver", "TFTDriver", "GifRecorder", "pick_driver",
    "FaceBridge", "attach_to_daemon", "build_face",
]


def build_face(width: int = 128, height: int = 64, driver: str = None,
               fps: int = 30, animations_dir: str = "animations", **kw):
    """Renderer + driver + player, wired and started. Returns the player."""
    renderer = FaceRenderer(width=width, height=height)
    drv = pick_driver(driver, **kw)
    player = FacePlayer(renderer=renderer, driver=drv, fps=fps,
                        animations_dir=animations_dir)
    return player
