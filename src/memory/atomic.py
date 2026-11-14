"""
src/memory/atomic.py

Writes that survive the power going out, and reads that survive what it left behind.

On a PC that froze or lost power 53 times in a month, ASH found `dwell.json` and `crashes.json.runs` empty at the next
start ("Dwell record unreadable; starting fresh") and lost them. Windows had renamed the temp file over the real one
before the data reached the disk, so the "atomic" replace published an empty file. A write that is meant to survive a crash
has to be: temp file in the same folder -> write -> flush -> fsync -> os.replace, and then the folder entry is safe.

    write_json(path, obj)      temp + fsync + replace, and the previous good copy kept as path + ".bak"
    write_text / write_bytes   the same for text and bytes
    write_with(path, fn)       the same for anything that writes to a file object (np.savez...)
    load_json(path, default)   the file, or else the ".bak", or else `default`; a file that will not parse is moved to
                               path + ".bad" (one is kept) and logged once, so the next start is not a repeat of the crash
    log_io_failure(logger, what, exc)
                               one warning a minute for a full disk or a locked file, with no traceback every second

Nothing here is clever, and everything that holds ASH's memory goes through it.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import threading
import time
from typing import Any, Callable, Dict, Optional

_last_logged: Dict[str, float] = {}
_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    """One lock per file: an autosave and the save on shutdown writing the same path must take turns."""
    key = os.path.abspath(path).lower()
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _replace(src: str, dst: str, tries: int = 6) -> None:
    """os.replace, trying again when Windows says the destination is busy (an antivirus scan or a reader has it open for a
    moment) -- the write used to be lost."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05 * (i + 1))


def is_disk_full(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and getattr(exc, "errno", None) in (errno.ENOSPC, errno.EDQUOT)


def log_io_failure(logger: logging.Logger, what: str, exc: BaseException, every_s: float = 60.0) -> None:
    """A failed write, said once a minute -- not a traceback on every tick while the disk is full."""
    now = time.time()
    if now - _last_logged.get(what, 0.0) < every_s:
        return
    _last_logged[what] = now
    if is_disk_full(exc):
        logger.error("Could not save %s: the disk is full (it will keep trying quietly)", what)
    elif isinstance(exc, OSError):
        logger.error("Could not save %s: %s", what, exc)
    else:
        logger.error("Could not save %s", what, exc_info=exc)


def write_with(path: str, writer: Callable[[Any], None], mode: str = "wb", keep_backup: bool = False,
               encoding: Optional[str] = None) -> None:
    """Write through `writer(fileobj)` to a temp file beside `path`, force it to disk, then swap it in."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"       # never a name another writer is using
    kw = {} if "b" in mode else {"encoding": encoding or "utf-8"}
    with _lock_for(path):
        try:
            with open(tmp, mode, **kw) as fh:
                writer(fh)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            if keep_backup:
                _keep_previous(path)
            _replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise


def _keep_previous(path: str) -> None:
    """The file as it is now becomes the backup -- as a second name for the same data (a hard link: free), so `path` is
    never missing in between, and only if it is not empty (an empty file must not replace the last good backup)."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        bak, hold = path + ".bak", f"{path}.{os.getpid()}.{threading.get_ident()}.bak.tmp"
        try:
            os.link(path, hold)
        except OSError:
            shutil.copyfile(path, hold)
        _replace(hold, bak)
    except OSError:
        logging.getLogger("ash.atomic").debug("could not keep a backup of %s", path, exc_info=True)


def write_bytes(path: str, data: bytes, keep_backup: bool = False) -> None:
    write_with(path, lambda f: f.write(data), "wb", keep_backup)


def write_text(path: str, text: str, keep_backup: bool = False, encoding: str = "utf-8") -> None:
    write_with(path, lambda f: f.write(text), "w", keep_backup, encoding)


def write_json(path: str, obj: Any, keep_backup: bool = True, **dump_kw) -> None:
    text = json.dumps(obj, **dump_kw)               # serialised first: a bad object never touches the file
    write_text(path, text, keep_backup)


def load_json(path: str, default: Any = None, logger: Optional[logging.Logger] = None) -> Any:
    """The parsed file; else its .bak; else `default`. An unparseable file is set aside as .bad."""
    log = logger or logging.getLogger("ash.atomic")
    for cand in (path, path + ".bak"):
        if not os.path.exists(cand):
            continue
        try:
            with open(cand, "r", encoding="utf-8") as fh:
                raw = fh.read()
            if not raw.strip():
                raise ValueError("empty file")
            return json.loads(raw)
        except ValueError as e:
            log.warning("%s is unreadable (%s)%s", cand, e, "; trying the backup" if cand == path else "")
            if cand == path:
                try:
                    os.replace(path, path + ".bad")          # damaged: set aside, so it is not a repeat of the crash
                except OSError:
                    pass
        except OSError as e:
            # Locked for a moment, a permissions blip: nothing is wrong with the file. Use the backup this once and leave it
            # where it is (it used to be moved aside, and the next save then overwrote the backup too).
            log.warning("%s could not be read just now (%s)%s", cand, e, "; using the backup" if cand == path else "")
    return default
