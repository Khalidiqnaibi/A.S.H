"""
src/daemon/samsungnotes.py

Samsung Notes, from ASH: read your notes, write a new one, add to an existing text note.

READING is from the app's own database, opened read-only (LocalState/Storage.sqlite under the Samsung Notes package):
titles, folders and the plain text the app keeps for its search (TextSearchDB). Nothing is ever written to that database.

WRITING is done the way a person would, through the app itself -- Samsung Notes has no API and its files are in a format ASH
must not touch (writing the database or the .sdocx files directly would corrupt notes and break sync). The app is driven
through Windows UI Automation by stable control ids (CreateNoteButton, Title, SearchBox, BackButton), with the clipboard for
the text, so the app saves and syncs it as its own. Every write is checked: afterwards the database is read again and the
note has to be there with the text in it. If it is not -- the app changed, the window was in the way -- ASH says so, and the
text is left on the clipboard (and in ASH's own notes folder), so the fix is one paste.

Safety:
  * writing needs a yes after ASH has read back what it will write, and says it takes the mouse for a few seconds;
  * "add to an existing note" only touches notes that are plain typed text -- never a PDF, an image or a handwritten note
    (document_extra says which), and only when exactly one note matches the name; otherwise it offers a new note instead;
  * the clipboard and the window that had focus are put back afterwards;
  * never for a guest's voice.

    "write this in Samsung Notes: buy milk"                  a new note (read back first)
    "new Samsung note called Lecture 5 saying ..."           a new note with a title
    "add ... to my NoSQL note in Samsung Notes"              added at the end of that note
    "start a Samsung note called Novel" ... "that's the note"  dictate it, then write it
    "read my last Samsung note" / "what's in my X Samsung note" / "find X in my Samsung notes" / "list my Samsung notes"

Rules and the clock only: no language model.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ash.daemon.samsungnotes")

PACKAGE_GLOB = r"%LOCALAPPDATA%\Packages\SAMSUNGELECTRONICSCoLtd.SamsungNotes_*"
APP_ID = "SAMSUNGELECTRONICSCoLtd.SamsungNotes_wyx1vj98g3asy!App"


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
class NotesDB:
    """The app's database, read-only."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or self.find()

    @staticmethod
    def find() -> str:
        for base in glob.glob(os.path.expandvars(PACKAGE_GLOB)):
            p = os.path.join(base, "LocalState", "Storage.sqlite")
            if os.path.exists(p):
                return p
        return ""

    @property
    def available(self) -> bool:
        return bool(self.path and os.path.exists(self.path))

    def _connect(self):
        uri = "file:" + self.path.replace("\\", "/") + "?mode=ro"
        try:
            return sqlite3.connect(uri, uri=True, timeout=3)
        except sqlite3.Error:
            # Locked by the app: read a copy (database and its write-ahead log), once.
            tmp = tempfile.mkdtemp(prefix="ash-sn-")
            for ext in ("", "-wal", "-shm"):
                if os.path.exists(self.path + ext):
                    shutil.copyfile(self.path + ext, os.path.join(tmp, "s.sqlite" + ext))
            return sqlite3.connect(os.path.join(tmp, "s.sqlite"))

    def _rows(self, sql: str, args: tuple = ()) -> List[tuple]:
        if not self.available:
            return []
        c = self._connect()
        try:
            return c.execute(sql, args).fetchall()
        finally:
            c.close()

    def count(self) -> int:
        r = self._rows("select count(*) from NoteDB where DeletedStatus=0")
        return int(r[0][0]) if r else 0

    def recent(self, n: int = 5) -> List[Dict[str, Any]]:
        rows = self._rows("select UUID, Title, CategoryUUID, LastModifiedAt, CreatedAt from NoteDB "
                          "where DeletedStatus=0 order by LastModifiedAt desc limit ?", (n,))
        return [self._note(r) for r in rows]

    def _note(self, r) -> Dict[str, Any]:
        return {"uuid": r[0], "title": r[1] or "", "folder": self.folder_name(r[2]), "modified": (r[3] or 0) / 1000.0,
                "created": (r[4] or 0) / 1000.0}

    def folder_name(self, category_uuid: str) -> str:
        r = self._rows("select DisplayName from CategoryTreeDB where UUID=?", (category_uuid,))
        if not r:
            r = self._rows("select DisplayName from CategoryDB where UUID=?", (category_uuid,))
        return (r[0][0] if r and r[0][0] else "")

    def in_folder(self, folder: str, limit: int = 12) -> List[Dict[str, Any]]:
        """Notes in the folder whose name holds `folder` (any folder, newest first)."""
        q = f"%{(folder or '').strip()}%"
        cats = self._rows("select UUID from CategoryTreeDB where DisplayName like ?", (q,)) \
            or self._rows("select UUID from CategoryDB where DisplayName like ?", (q,))
        out: List[Dict[str, Any]] = []
        for (cu,) in cats:
            rows = self._rows("select UUID, Title, CategoryUUID, LastModifiedAt, CreatedAt from NoteDB "
                              "where DeletedStatus=0 and CategoryUUID=? order by LastModifiedAt desc limit ?", (cu, limit))
            out += [self._note(r) for r in rows]
        return out[:limit]

    def find_title(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        q = f"%{(query or '').strip()}%"
        rows = self._rows("select UUID, Title, CategoryUUID, LastModifiedAt, CreatedAt from NoteDB "
                          "where DeletedStatus=0 and Title like ? order by LastModifiedAt desc limit ?", (q, limit))
        return [self._note(r) for r in rows]

    def search(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Notes whose text (typed, or recognised from handwriting) or title holds the words."""
        words = [w for w in re.findall(r"\w+", query or "") if len(w) > 1]
        if not words:
            return []
        cond = " and ".join(["(coalesce(t.StrippedContent,'') || ' ' || coalesce(t.HWTextContent,'') || ' ' || "
                             "coalesce(n.Title,'')) like ?"] * len(words))
        rows = self._rows("select n.UUID, n.Title, n.CategoryUUID, n.LastModifiedAt, n.CreatedAt from NoteDB n "
                          "left join TextSearchDB t on t.UUID = n.UUID where n.DeletedStatus=0 and " + cond +
                          " order by n.LastModifiedAt desc limit ?", tuple(f"%{w}%" for w in words) + (limit,))
        return [self._note(r) for r in rows]

    def text_of(self, uuid: str) -> str:
        r = self._rows("select coalesce(StrippedContent,''), coalesce(HWTextContent,''), coalesce(PDFTextContents,'') "
                       "from TextSearchDB where UUID=?", (uuid,))
        if not r:
            return ""
        return " ".join(x.strip() for x in r[0] if x and x.strip())

    def kind_of(self, uuid: str) -> Dict[str, bool]:
        r = self._rows("select hasSticky, hasImage, hasStroke, hasPdf from document_extra where UUID=?", (uuid,))
        if not r:
            return {"text_only": False, "known": False}
        s, i, k, p = (bool(x) for x in r[0])
        return {"text_only": not (s or i or k or p), "known": True, "sticky": s, "image": i, "stroke": k, "pdf": p}


# ----------------------------------------------------------------------------
# Driving the app
# ----------------------------------------------------------------------------
class Driver:
    """Windows UI Automation + clipboard + a few keys. Everything the app needs, nothing it does not."""

    def __init__(self):
        self._uia = None
        self._UIA = None
        self.launched_by_us = False

    # -- plumbing -------------------------------------------------------------
    def available(self) -> bool:
        if os.name != "nt":
            return False
        try:
            self._load()
            return True
        except Exception:
            return False

    def _load(self):
        if self._uia is not None:
            return
        import comtypes.client
        comtypes.client.GetModule("UIAutomationCore.dll")
        from comtypes.gen import UIAutomationClient as UIA
        self._UIA = UIA
        self._uia = comtypes.client.CreateObject("{ff48dba4-60ef-4201-aa87-54103eef594e}", interface=UIA.IUIAutomation)
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()

    def window(self):
        self._load()
        UIA = self._UIA
        ws = self._uia.GetRootElement().FindAll(UIA.TreeScope_Children, self._uia.CreatePropertyCondition(
            UIA.UIA_NamePropertyId, "Samsung Notes"))
        if not ws or not ws.Length:
            return None
        e = ws.GetElement(0)
        return e if e else None

    def by_id(self, aid: str, root=None):
        UIA = self._UIA
        w = root or self.window()
        if w is None:
            return None
        e = w.FindFirst(UIA.TreeScope_Descendants, self._uia.CreatePropertyCondition(UIA.UIA_AutomationIdPropertyId, aid))
        return e if e else None                      # comtypes hands back a NULL pointer object, not None

    def by_name(self, name: str, control_type: Optional[int] = None, root=None):
        UIA = self._UIA
        w = root or self.window()
        if w is None:
            return None
        cond = self._uia.CreatePropertyCondition(UIA.UIA_NamePropertyId, name)
        if control_type is not None:
            cond = self._uia.CreateAndCondition(cond, self._uia.CreatePropertyCondition(UIA.UIA_ControlTypePropertyId, control_type))
        e = w.FindFirst(UIA.TreeScope_Descendants, cond)
        return e if e else None

    def items_named(self, contains: str) -> List[Any]:
        UIA = self._UIA
        w = self.window()
        out = []
        if w is None:
            return out
        els = w.FindAll(UIA.TreeScope_Descendants, self._uia.CreatePropertyCondition(UIA.UIA_ControlTypePropertyId, 50007))
        for i in range(els.Length):
            e = els.GetElement(i)
            try:
                if contains.lower() in (e.CurrentName or "").lower():
                    out.append(e)
            except Exception:
                pass
        return out

    def center(self, el) -> Tuple[int, int]:
        r = el.CurrentBoundingRectangle
        return (r.left + r.right) // 2, (r.top + r.bottom) // 2

    def invoke(self, el) -> bool:
        UIA = self._UIA
        try:
            el.GetCurrentPattern(UIA.UIA_InvokePatternId).QueryInterface(UIA.IUIAutomationInvokePattern).Invoke()
            return True
        except Exception:
            x, y = self.center(el)
            self.click(x, y)
            return True

    def select(self, el) -> bool:
        UIA = self._UIA
        try:
            el.GetCurrentPattern(UIA.UIA_SelectionItemPatternId).QueryInterface(UIA.IUIAutomationSelectionItemPattern).Select()
            return True
        except Exception:
            return False

    # -- the app --------------------------------------------------------------
    def launch(self, wait_s: float = 25.0) -> bool:
        import subprocess
        if self.window() is None:
            subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{APP_ID}"])
            self.launched_by_us = True
        end = time.time() + wait_s
        while time.time() < end:
            if self.window() is not None:
                time.sleep(1.5)
                self.dismiss_dialog()
                return True
            time.sleep(0.5)
        return False

    def dismiss_dialog(self):
        # The "optimised for Galaxy Books" notice blocks every other control until it is answered.
        ok = self.by_name("OK", 50000)
        if ok is not None:
            try:
                self.invoke(ok)
                time.sleep(0.8)
            except Exception:
                pass

    def to_list(self, tries: int = 5) -> bool:
        """Back out of a note and of search results until the note list (with its Create note button) is showing."""
        for _ in range(tries):
            if self.by_id("CreateNoteButton") is not None:
                return True
            back = (self.by_id("BackButton") or self.by_id("FilterBackButton"))
            if back is None:
                break
            self.invoke(back)
            time.sleep(1.5)
        return self.by_id("CreateNoteButton") is not None

    def wait_for(self, aid: str, seconds: float = 8.0):
        end = time.time() + seconds
        while time.time() < end:
            e = self.by_id(aid)
            if e is not None:
                return e
            time.sleep(0.3)
        return None

    def close_app(self):
        import ctypes
        w = self.window()
        if w is not None:
            ctypes.windll.user32.PostMessageW(w.CurrentNativeWindowHandle, 0x0010, 0, 0)       # WM_CLOSE

    # -- input ----------------------------------------------------------------
    def click(self, x: int, y: int):
        import ctypes
        u = ctypes.windll.user32
        u.SetCursorPos(int(x), int(y))
        time.sleep(0.15)
        u.mouse_event(2, 0, 0, 0, 0)
        u.mouse_event(4, 0, 0, 0, 0)
        time.sleep(0.3)

    def key(self, vk: int, ctrl: bool = False):
        import ctypes
        u = ctypes.windll.user32
        if ctrl:
            u.keybd_event(0x11, 0, 0, 0)
        u.keybd_event(vk, 0, 0, 0)
        u.keybd_event(vk, 0, 2, 0)
        if ctrl:
            u.keybd_event(0x11, 0, 2, 0)
        time.sleep(0.4)

    def paste(self, text: str):
        self.clipboard_set(text)
        self.key(0x56, ctrl=True)
        time.sleep(0.4)

    def foreground(self) -> int:
        import ctypes
        return int(ctypes.windll.user32.GetForegroundWindow() or 0)

    def restore_foreground(self, hwnd: int):
        import ctypes
        if hwnd:
            try:
                ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception:
                pass

    def clipboard_get(self) -> Optional[str]:
        import ctypes
        u, k = ctypes.windll.user32, ctypes.windll.kernel32
        u.GetClipboardData.restype = ctypes.c_void_p
        k.GlobalLock.restype = ctypes.c_void_p
        k.GlobalLock.argtypes = [ctypes.c_void_p]
        k.GlobalUnlock.argtypes = [ctypes.c_void_p]
        if not u.OpenClipboard(None):
            return None
        try:
            h = u.GetClipboardData(13)
            if not h:
                return None
            p = k.GlobalLock(h)
            try:
                return ctypes.wstring_at(p)
            finally:
                k.GlobalUnlock(h)
        finally:
            u.CloseClipboard()

    def clipboard_set(self, text: str):
        import ctypes
        u, k = ctypes.windll.user32, ctypes.windll.kernel32
        k.GlobalAlloc.restype = ctypes.c_void_p
        k.GlobalLock.restype = ctypes.c_void_p
        k.GlobalLock.argtypes = [ctypes.c_void_p]
        k.GlobalUnlock.argtypes = [ctypes.c_void_p]
        u.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        data = (text + "\0").encode("utf-16-le")
        for _ in range(20):
            if u.OpenClipboard(None):
                break
            time.sleep(0.05)
        u.EmptyClipboard()
        h = k.GlobalAlloc(0x0042, len(data))
        p = k.GlobalLock(h)
        ctypes.memmove(p, data, len(data))
        k.GlobalUnlock(h)
        u.SetClipboardData(13, h)
        u.CloseClipboard()


# ----------------------------------------------------------------------------
# What is said
# ----------------------------------------------------------------------------
_LEAD = r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:(?:please|can\s+you|could\s+you)[,\s]+)*"
_SN = r"samsung\s+notes?"
_TITLE = r"(?:called|titled|named|with\s+the\s+title)\s+(?P<title>[^:,]{1,60}?)"
_WRITE_TO = re.compile(_LEAD + r"(?:write|put|save|type|add|make|jot|note|log)\s+(?:down\s+)?(?:this|that|the\s+following|it)?[,:\s]*"
                       r"(?P<body>.*?)\s*(?:in|into|to|on)\s+(?:my\s+)?(?:a\s+)?(?:new\s+)?" + _SN +
                       r"(?:\s+" + _TITLE + r")?\s*(?:[:,]\s*(?P<body2>.+))?\s*[.!?]*\s*$", re.I)
_NEW = re.compile(_LEAD + r"(?:(?:make|create|start|open|write|take)\s+(?:me\s+)?)?(?:a\s+)?new\s+" + _SN + r"(?:\s+note)?"
                  r"(?:\s+" + _TITLE + r")?(?:\s+(?:saying|with|that\s+says|containing)[:,]?\s+(?P<body>.+?))?\s*[.!?]*\s*$", re.I)
_APPEND = re.compile(_LEAD + r"(?:add|append|write|put)\s+(?P<body>.+?)\s+(?:to|onto|in|into)\s+(?:the\s+end\s+of\s+)?(?:my\s+)?"
                     r"(?P<t>[\w .'\-]{2,60}?)\s+(?:note\s+)?(?:in|on)\s+" + _SN + r"\s*[.!?]*\s*$", re.I)
_APPEND2 = re.compile(_LEAD + r"(?:add|append)\s+(?:to|onto)\s+(?:my\s+)?(?P<t>[\w .'\-]{2,60}?)\s+" + _SN +
                      r"(?:\s+note)?\s*[:,]\s*(?P<body>.+?)\s*[.!?]*\s*$", re.I)
_START = re.compile(_LEAD + r"(?:start|begin|open)\s+(?:dictating|a\s+dictation|taking\s+(?:a\s+)?note|writing)\s+(?:a\s+|an\s+|the\s+)?"
                    r"(?:" + _SN + r"|note)?(?:\s+note)?(?:\s+" + _TITLE + r")?\s*[.!?]*\s*$", re.I)
_START2 = re.compile(_LEAD + r"start\s+(?:a\s+)?" + _SN + r"(?:\s+note)?(?:\s+" + _TITLE + r")?\s*[.!?]*\s*$", re.I)
_END = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:that'?s\s+(?:the|my)\s+note|that'?s\s+it|save\s+(?:the\s+|my\s+)?note|"
                  r"end\s+(?:the\s+)?note|done\s+(?:writing|dictating)|stop\s+(?:dictating|writing)|finish\s+(?:the\s+)?note)\s*[.!?]*\s*$", re.I)
_DROP = re.compile(r"^\s*(?:(?:hey\s+)?ash[,\s]+)?(?:cancel|discard|scratch|forget|drop)\s+(?:the\s+|that\s+|this\s+)?(?:note|dictation)\s*[.!?]*\s*$", re.I)
_READ_LAST = re.compile(_LEAD + r"(?:read|show|open(?:\s+up)?|what'?s\s+in)\s+(?:me\s+)?(?:my\s+|the\s+)?(?:last|latest|newest|most\s+recent)\s+note\s+(?:in|on|from|of|and|into)\s+" + _SN +
                        r"\s*[.!?]*\s*$|" + _LEAD + r"(?:read|show|open(?:\s+up)?|what'?s\s+in)\s+(?:me\s+)?(?:my\s+)?(?:last|latest|newest|most\s+recent)\s+" + _SN +
                        r"(?:\s+note)?\s*[.!?]*\s*$|" + _LEAD + r"what\s+(?:was|is)\s+my\s+(?:last|latest)\s+" + _SN + r"(?:\s+note)?\s*\??\s*$", re.I)
_READ = re.compile(_LEAD + r"(?:read|what'?s\s+in|what\s+does|tell\s+me\s+what'?s\s+in)\s+(?:me\s+)?(?:my\s+)?(?P<t>[\w .'\-]{2,60}?)\s+" + _SN +
                   r"(?:\s+note)?(?:\s+say)?\s*[.!?]*\s*$", re.I)
_FIND = re.compile(_LEAD + r"(?:find|search(?:\s+for)?|look\s+for|look\s+up|what\s+did\s+i\s+write\s+about)\s+(?P<q>.+?)\s+(?:in|on|through)\s+(?:my\s+)?" + _SN +
                   r"\s*\??\s*[.!?]*\s*$", re.I)
_FIND2 = re.compile(_LEAD + r"(?:in|on|through)\s+(?:my\s+)?" + _SN + r"[,\s]+(?:please\s+)?(?:find|search(?:\s+for)?|look\s+for|look\s+up)\s+(?P<q>.+?)\s*[.!?]*\s*$", re.I)
_LIST = re.compile(_LEAD + r"(?:list|what|which|show)\s+(?:me\s+)?(?:my\s+)?(?:recent\s+)?" + _SN +
                   r"(?:\s+(?:do\s+i\s+have|are\s+there))?\s*\??\s*[.!?]*\s*$|" + _LEAD + r"(?:what|which|list|show)\s+(?:me\s+)?(?:my\s+)?(?:the\s+)?notes?\s+(?:do\s+i\s+have|are\s+there)\s+"
                   r"(?:in|on)\s+(?:my\s+)?" + _SN + r"\s*\??\s*[.!?]*\s*$", re.I)


_FOLDER = re.compile(_LEAD + r"(?:what|which|list|show|tell\s+me)\s+(?:me\s+)?(?:the\s+)?(?:my\s+)?(?:notes?\s+)?(?:do\s+i\s+have\s+|are\s+there\s+)?"
                     r"(?:notes?\s+)?(?:in|inside|under|from)\s+(?:the\s+|my\s+)?(?P<f>[\w .'\-]{1,40}?)\s+folder(?:\s+(?:in|on|of)\s+(?:my\s+)?" + _SN +
                     r")?\s*\??\s*[.!?]*\s*$|" + _LEAD + r"(?:what|which|list|show)\s+(?:me\s+)?(?:the\s+)?(?:my\s+)?notes?\s+(?:do\s+i\s+have\s+)?"
                     r"(?:in|inside|under)\s+(?:the\s+|my\s+)?(?P<f2>[\w .'\-]{1,40}?)\s+(?:folder\s+)?(?:in|on|of)\s+(?:my\s+)?" + _SN +
                     r"\s*\??\s*[.!?]*\s*$", re.I)
_CAN = re.compile(_LEAD + r"(?:what\s+(?:can|do)\s+you\s+(?:do|know)\s+(?:with|about|in)|what\s+are\s+you\s+able\s+to\s+do\s+(?:with|in)|can\s+you\s+(?:use|work\s+with|write\s+(?:in|to|into)|read|access)|do\s+you\s+(?:have|support|work\s+with)|are\s+you\s+(?:able\s+to\s+use|connected\s+to))"
                  r"\s+(?:my\s+)?" + _SN + r"\s*\?\s*$", re.I)
CAPABILITY = ("With Samsung Notes I can read your notes (the last one, one by title, a search, the list, or what's in a folder) from the "
              "app's own database, and write: a new note, or text added to the end of a plain text note. Writing takes over the mouse "
              "for a few seconds and I read it back and wait for a yes first.")


def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip(" \t\"'“”.,;:")).strip()


def _say_list(titles: List[str]) -> str:
    t = [re.sub(r"_[0-9a-f]{20,}$", "", x) for x in titles if x]
    if len(t) <= 1:
        return "".join(t)
    return ", ".join(t[:-1]) + " and " + t[-1]


class SamsungNotes:
    PENDING_S = 90.0

    def __init__(self, db: Optional[NotesDB] = None, driver: Optional[Driver] = None, enabled: bool = True,
                 write: bool = True, clock: Callable[[], float] = time.time, backup_dir: Optional[str] = None,
                 notify: Optional[Callable[[str], None]] = None, run_async: bool = True):
        self.db = db or NotesDB()
        self.driver = driver or Driver()
        self.enabled = enabled
        self.write = write
        self.clock = clock
        self.backup_dir = backup_dir
        self.notify = notify                       # called with what to say when a background write finishes
        self.run_async = run_async
        self.pending: Optional[Dict[str, Any]] = None
        self.dictating: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self.stats = {"read": 0, "written": 0, "appended": 0, "failed": 0}

    # -- hearing --------------------------------------------------------------------------------------
    def mentions(self, text: str) -> bool:
        return bool(re.search(r"samsung\s+notes?", text or "", re.I))

    def hear(self, text: str, now: Optional[float] = None) -> Optional[str]:
        """What to say, or None if this is not about Samsung Notes."""
        if not self.enabled:
            return None
        now = self.clock() if now is None else now
        t = (text or "").strip()
        if not t or len(t) > 1500:
            return None

        # answering a read-back
        p = self.pending
        if p is not None:
            if now - p["t"] > self.PENDING_S:
                self.pending = None
            else:
                from . import yesno
                if yesno.is_yes(t, strong=True) or yesno.is_yes(t):
                    self.pending = None
                    return self._run(p)
                if yesno.is_no(t):
                    self.pending = None
                    return "Okay, I won't write it."

        # dictation in progress
        d = self.dictating
        if d is not None:
            if now - d["last"] > 15 * 60:
                self.dictating = None
            elif _DROP.match(t):
                self.dictating = None
                return "Okay, I've dropped that note."
            elif _END.match(t):
                self.dictating = None
                if not d["parts"]:
                    return "There was nothing in it, so I haven't written anything."
                return self._ask({"kind": "new", "title": d["title"], "body": "\n".join(d["parts"])}, now)
            else:
                d["parts"].append(t)
                d["last"] = now
                return ""              # taken down quietly: the point is not to be interrupted while writing

        if not self.mentions(t) and not _START.match(t):
            return None

        if _CAN.match(t):
            return CAPABILITY + ("" if self.write else " (Writing is switched off right now.)")

        m = _START.match(t) or _START2.match(t)
        if m:
            self.dictating = {"title": _clean(m.groupdict().get("title")), "parts": [], "last": now}
            return ("Go ahead -- I'll take it down" + (f" as \"{self.dictating['title']}\"" if self.dictating["title"] else "") +
                    ". Say \"that's the note\" when you're done.")

        m = _NEW.match(t)
        if m:
            body = _clean(m.group("body"))
            if body:
                return self._ask({"kind": "new", "title": _clean(m.group("title")), "body": body}, now)
            self.dictating = {"title": _clean(m.group("title")), "parts": [], "last": now}
            return "Go ahead -- I'll take it down. Say \"that's the note\" when you're done."

        m = _APPEND.match(t) or _APPEND2.match(t)
        if m:
            return self._ask({"kind": "append", "title": _clean(m.group("t")), "body": _clean(m.group("body"))}, now)

        m = _WRITE_TO.match(t)
        if m:
            body = _clean(m.group("body")) or _clean(m.groupdict().get("body2"))
            if m.groupdict().get("body2") and not _clean(m.group("body")):
                body = _clean(m.group("body2"))
            if not body or body.lower() in ("this", "that", "it"):
                self.dictating = {"title": _clean(m.group("title")), "parts": [], "last": now}
                return "Go ahead -- I'll take it down. Say \"that's the note\" when you're done."
            return self._ask({"kind": "new", "title": _clean(m.group("title")), "body": body}, now)

        m = _FOLDER.match(t)
        if m:
            return self._folder(_clean(m.group("f") or m.group("f2")))
        if _READ_LAST.match(t):
            return self._read_last()
        m = _FIND.match(t) or _FIND2.match(t)
        if m:
            return self._find(_clean(m.group("q")))
        m = _READ.match(t)
        if m:
            return self._read(_clean(m.group("t")))
        if _LIST.match(t):
            return self._list()
        return None

    # -- reading --------------------------------------------------------------------------------------
    def _need_db(self) -> Optional[str]:
        if not self.db.available:
            return "I can't find Samsung Notes on this computer."
        return None

    def _speak_text(self, s: str, n: int = 420) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "..."

    def _read_last(self) -> str:
        if (m := self._need_db()):
            return m
        self.stats["read"] += 1
        r = self.db.recent(1)
        if not r:
            return "You don't have any Samsung notes yet."
        n = r[0]
        text = self.db.text_of(n["uuid"])
        title = re.sub(r"_[0-9a-f]{20,}$", "", n["title"])
        return (f"Your last note is \"{title}\": {self._speak_text(text)}" if text
                else f"Your last note is \"{title}\", but it has no text I can read -- it may be handwriting or a PDF.")

    def _read(self, title: str) -> str:
        if (m := self._need_db()):
            return m
        self.stats["read"] += 1
        found = self.db.find_title(title)
        if not found:
            found = self.db.search(title)
        if not found:
            return f"I can't find a Samsung note called {title}."
        if len(found) > 1 and not any(f["title"].lower() == title.lower() for f in found):
            return "Which one -- " + _say_list([f["title"] for f in found[:4]]) + "?"
        n = next((f for f in found if f["title"].lower() == title.lower()), found[0])
        text = self.db.text_of(n["uuid"])
        clean = re.sub(r"_[0-9a-f]{20,}$", "", n["title"])
        return (f"\"{clean}\": {self._speak_text(text)}" if text else f"\"{clean}\" has no text I can read -- it may be handwriting or a PDF.")

    def _folder(self, name: str) -> str:
        if (m := self._need_db()):
            return m
        self.stats["read"] += 1
        found = self.db.in_folder(name)
        if not found:
            return f"I can't find any notes in a Samsung Notes folder called {name}."
        folder = found[0].get("folder") or name
        titles = _say_list([f["title"] for f in found[:8]])
        more = "" if len(found) <= 8 else f" and {len(found) - 8} more"
        return f"In {folder} you have {len(found)} note{'s' if len(found) != 1 else ''}: {titles}{more}."

    def _find(self, q: str) -> str:
        if (m := self._need_db()):
            return m
        self.stats["read"] += 1
        found = self.db.search(q)
        if not found:
            return f"I didn't find anything about {q} in your Samsung notes."
        names = _say_list([f["title"] for f in found[:4]])
        return f"{len(found)} note{'s' if len(found) != 1 else ''} mention{'' if len(found) != 1 else 's'} {q}: {names}."

    def _list(self) -> str:
        if (m := self._need_db()):
            return m
        self.stats["read"] += 1
        r = self.db.recent(5)
        if not r:
            return "You don't have any Samsung notes yet."
        return f"You have {self.db.count()} notes. The latest are " + _say_list([x["title"] for x in r]) + "."

    # -- writing --------------------------------------------------------------------------------------
    def _ask(self, job: Dict[str, Any], now: float) -> str:
        if not self.write:
            return "Writing to Samsung Notes is switched off (samsung_notes.write)."
        if (m := self._need_db()):
            return m
        if not self.driver.available():
            return "I can't drive Samsung Notes here -- I'd need Windows UI Automation (comtypes)."
        body = job["body"]
        if len(body) > 6000:
            return "That's too long to type in one go -- say it in parts."
        if job["kind"] == "append":
            found = self.db.find_title(job["title"])
            exact = [f for f in found if re.sub(r"_[0-9a-f]{20,}$", "", f["title"]).lower() == job["title"].lower()]
            cand = exact or found
            if not cand:
                return f"I can't find a Samsung note called {job['title']}. Say \"new Samsung note\" and I'll start one."
            if len(cand) > 1:
                return "Which one -- " + _say_list([c["title"] for c in cand[:4]]) + "?"
            kind = self.db.kind_of(cand[0]["uuid"])
            if not kind.get("text_only"):
                what = "a PDF" if kind.get("pdf") else "handwritten" if kind.get("stroke") else "a note with pictures" if kind.get("image") else "not a plain text note"
                return (f"\"{_say_list([cand[0]['title']])}\" is {what}, and I won't write into it in case I spoil it. "
                        f"Say \"new Samsung note called ...\" and I'll make a new one beside it.")
            job = {**job, "uuid": cand[0]["uuid"], "title": cand[0]["title"]}
        job["t"] = now
        self.pending = job
        shown = self._speak_text(body, 120)
        if job["kind"] == "new":
            where = f"a new Samsung note{' called ' + job['title'] if job['title'] else ''}"
        else:
            where = f"the end of \"{_say_list([job['title']])}\""
        return f"I'll put \"{shown}\" in {where}. It takes over the mouse for a few seconds. Say yes to go ahead."

    def _run(self, job: Dict[str, Any]) -> str:
        def work():
            try:
                res = self.write_new(job["title"], job["body"]) if job["kind"] == "new" else self.append(job["uuid"], job["title"], job["body"])
            except Exception:
                logger.exception("Writing to Samsung Notes failed")
                res = {"ok": False, "verified": False, "why": "it raised an error"}
            say = self._outcome(job, res)
            if self.notify is not None:
                try:
                    self.notify(say)
                except Exception:
                    logger.exception("Could not say what happened to the note")
            return say
        if self.run_async:
            threading.Thread(target=work, daemon=True, name="ash-samsung-notes").start()
            return "On it."
        return work()

    def _outcome(self, job: Dict[str, Any], res: Dict[str, Any]) -> str:
        if res.get("ok") and res.get("verified"):
            self.stats["appended" if job["kind"] == "append" else "written"] += 1
            if job["kind"] == "append":
                return f"Added to \"{_say_list([job['title']])}\"."
            folder = f" in {res['folder']}" if res.get("folder") else ""
            return f"Done -- it's in Samsung Notes{folder} as \"{res.get('title') or job['title'] or 'a new note'}\"."
        self.stats["failed"] += 1
        self._backup(job)
        why = res.get("why") or "I couldn't confirm it was saved"
        return (f"I couldn't get that into Samsung Notes -- {why}. The text is on your clipboard"
                f"{' and in my notes folder' if self.backup_dir else ''}, so it's one paste away.")

    def _backup(self, job: Dict[str, Any]):
        try:
            self.driver.clipboard_set(job["body"])
        except Exception:
            pass
        if self.backup_dir:
            try:
                os.makedirs(self.backup_dir, exist_ok=True)
                name = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(self.clock())) + ".txt"
                with open(os.path.join(self.backup_dir, name), "w", encoding="utf-8") as fh:
                    fh.write((job.get("title") or "") + "\n\n" + job["body"])
            except OSError:
                logger.warning("Could not keep a copy of the note text")

    def write_new(self, title: str, body: str) -> Dict[str, Any]:
        d, db = self.driver, self.db
        title = title or "Note from ASH " + time.strftime("%d %b %H:%M", time.localtime(self.clock()))
        started = self.clock() * 1000.0 - 5000
        prev_clip, prev_fg = None, 0
        try:
            prev_clip, prev_fg = d.clipboard_get(), d.foreground()
        except Exception:
            pass
        try:
            if not d.launch():
                return {"ok": False, "verified": False, "why": "the app would not open"}
            d.dismiss_dialog()
            if not d.to_list():                      # out of whatever note or search is open, to the list
                return {"ok": False, "verified": False, "why": "I couldn't get back to its list of notes"}
            allnotes = d.by_name("All notes", 50007)
            if allnotes is not None:
                d.select(allnotes) or d.click(*d.center(allnotes))
                time.sleep(1.0)
            btn = d.wait_for("CreateNoteButton", 6)
            if btn is None:
                return {"ok": False, "verified": False, "why": "I couldn't find its Create note button"}
            d.click(*d.center(btn))      # a real click: this button's Invoke is accepted and ignored
            t = d.wait_for("Title", 8)
            if t is None:
                return {"ok": False, "verified": False, "why": "the new note didn't open"}
            try:
                t.SetFocus()                         # focus it for real; a click on its centre can land on the page below
            except Exception:
                pass
            d.click(*d.center(t))
            time.sleep(0.4)
            d.paste(title)
            d.key(0x0D)                              # Enter: from the title to the text
            d.paste(body)
            time.sleep(1.2)
            back = d.by_id("BackButton")
            if back is not None:
                d.invoke(back)                       # leaving the note is what saves it
            return self._verify_new(title, body, started)
        finally:
            self._tidy(prev_clip, prev_fg)

    def _verify_new(self, title: str, body: str, started_ms: float) -> Dict[str, Any]:
        snippet = re.sub(r"\s+", " ", body)[:30]
        end = time.time() + 15
        while time.time() < end:
            for n in self.db.recent(5):
                if n["title"] == title and n["created"] * 1000.0 >= started_ms:
                    txt = re.sub(r"\s+", " ", self.db.text_of(n["uuid"]))
                    if snippet in txt:
                        return {"ok": True, "verified": True, "title": title, "folder": n["folder"], "uuid": n["uuid"]}
            time.sleep(1.0)
        return {"ok": True, "verified": False, "why": "the app didn't show it in its list afterwards"}

    def append(self, uuid: str, title: str, body: str) -> Dict[str, Any]:
        d = self.driver
        kind = self.db.kind_of(uuid)
        if not kind.get("text_only"):
            return {"ok": False, "verified": False, "why": "it isn't a plain text note"}
        before = re.sub(r"\s+", " ", self.db.text_of(uuid))
        prev_clip, prev_fg = None, 0
        try:
            prev_clip, prev_fg = d.clipboard_get(), d.foreground()
        except Exception:
            pass
        try:
            if not d.launch():
                return {"ok": False, "verified": False, "why": "the app would not open"}
            d.dismiss_dialog()
            if not d.to_list():
                return {"ok": False, "verified": False, "why": "I couldn't get back to its list of notes"}
            box = d.by_id("SearchBox")
            if box is None:
                return {"ok": False, "verified": False, "why": "I couldn't find its search box"}
            d.click(*d.center(box))
            d.paste(re.sub(r"_[0-9a-f]{20,}$", "", title))
            d.key(0x0D)
            time.sleep(2.0)
            hits = d.items_named(re.sub(r"_[0-9a-f]{20,}$", "", title)[:30])
            if len(hits) != 1:
                return {"ok": False, "verified": False, "why": f"the search showed {len(hits)} notes, not one"}
            d.click(*d.center(hits[0]))
            ed = d.wait_for("SpenComposerViewScrollViewer", 8)
            if ed is None:
                return {"ok": False, "verified": False, "why": "the note didn't open"}
            x, y = d.center(ed)
            d.click(x, y - 120)                      # inside the page, near the top: focus the text
            d.key(0x23, ctrl=True)                   # Ctrl+End: the end of the note
            d.key(0x0D)
            d.paste(body)
            time.sleep(1.2)
            back = d.by_id("BackButton")
            if back is not None:
                d.invoke(back)
            snippet = re.sub(r"\s+", " ", body)[:30]
            end = time.time() + 15
            while time.time() < end:
                after = re.sub(r"\s+", " ", self.db.text_of(uuid))
                if snippet in after and after != before and before.split(" ")[:3] == after.split(" ")[:3]:
                    return {"ok": True, "verified": True}
                time.sleep(1.0)
            return {"ok": True, "verified": False, "why": "the note didn't show the new text afterwards"}
        finally:
            self._tidy(prev_clip, prev_fg)

    def _tidy(self, prev_clip: Optional[str], prev_fg: int):
        d = self.driver
        try:
            d.to_list()                              # leave the app on its list, not inside a note or a search
        except Exception:
            pass
        try:
            if d.launched_by_us:
                d.close_app()
                d.launched_by_us = False
        except Exception:
            pass
        try:
            if prev_clip is not None:
                d.clipboard_set(prev_clip)           # what you had copied is back
            d.restore_foreground(prev_fg)
        except Exception:
            pass

    # -- managing -------------------------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "found": self.db.available, "write": self.write, "notes": self.db.count() if self.db.available else 0,
                "dictating": self.dictating is not None, "waiting": self.pending is not None, **self.stats}
