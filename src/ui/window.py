"""
src/ui/window.py

Open the console as its own window rather than a browser tab.

Chromium browsers (Edge ships with every Windows install) have an app mode,
`--app=URL`: no tabs, no address bar, its own taskbar entry -- it looks like
a native chat window and costs no new dependency. A separate profile
directory keeps it out of the user's real browser session (history, cookies,
extensions). If no Chromium browser is found the console still opens, just
as an ordinary tab.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import webbrowser
from typing import List, Optional

logger = logging.getLogger("ash.ui.window")


def _candidates() -> List[str]:
    if sys.platform == "win32":
        roots = [os.environ.get(k) for k in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
        rel = [r"Microsoft\Edge\Application\msedge.exe",
               r"Google\Chrome\Application\chrome.exe",
               r"BraveSoftware\Brave-Browser\Application\brave.exe"]
        return [os.path.join(r, p) for p in rel for r in roots if r]
    if sys.platform == "darwin":
        return ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                "/Applications/Chromium.app/Contents/MacOS/Chromium"]
    return [p for p in (shutil.which(n) for n in
            ("google-chrome", "chromium", "chromium-browser", "microsoft-edge")) if p]


def find_browser() -> Optional[str]:
    for p in _candidates():
        if p and os.path.isfile(p):
            return p
    return None


def open_window(url: str, profile_dir: Optional[str] = None,
                size=(460, 800)) -> str:
    """Open the console. Returns how: 'app', 'tab' or 'none'."""
    exe = find_browser()
    if exe:
        args = [exe, f"--app={url}", f"--window-size={size[0]},{size[1]}",
                "--no-first-run", "--no-default-browser-check"]
        if profile_dir:
            os.makedirs(profile_dir, exist_ok=True)
            args.append(f"--user-data-dir={os.path.abspath(profile_dir)}")
        try:
            flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, close_fds=True, creationflags=flags)
            return "app"
        except Exception:
            logger.warning("Could not start %s in app mode", exe, exc_info=True)
    try:
        return "tab" if webbrowser.open(url) else "none"
    except Exception:
        return "none"
