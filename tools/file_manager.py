# tools/file_manager.py
"""
Deterministic, extensible file-operations layer for ASH.

Same philosophy as calculator_tool/date_time_tool and the emotion engine
in this codebase: every function here is plain code -- os.stat calls,
parsing, regex -- nothing in this module ever calls an LLM or guesses.
Each operation returns a structured dict of facts; the LLM's only job
(in ash.py's _render_with_llm) is to phrase those facts for the user.

Adding support for a new file type means writing a small handler and
calling register_handler() -- the dispatcher (FileTypeRegistry, and the
read/search/info functions below) never needs to change. That's the
"different and expanding amount of file types" part: extensions register
themselves, nobody edits a growing if/elif chain.
"""

import csv
import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ash.files")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(stream=sys.stderr))

# Resource guards -- never let one file operation balloon memory or the
# eventual LLM prompt, regardless of how big the file/folder actually is.
MAX_PREVIEW_CHARS = int(os.environ.get("ASH_FILE_MAX_PREVIEW_CHARS", 8000))
MAX_LIST_ENTRIES = 200
MAX_SEARCH_RESULTS = 25
MAX_SEARCH_FILES_SCANNED = 500

def _error(op: str, path: str, message: str, **extra) -> Dict[str, Any]:
    logger.warning("[%s] %s -> %s", op, path, message)
    return {"ok": False, "operation": op, "path": path, "error": message, **extra}


def _ok(op: str, path: str, **data) -> Dict[str, Any]:
    return {"ok": True, "operation": op, "path": path, **data}


@dataclass
class FileHandlerResult:
    kind: str                                    # "text", "json", "csv", "image", "pdf", "binary", ...
    preview: Optional[str] = None                # capped, human-readable text preview
    structure: Dict[str, Any] = field(default_factory=dict)  # type-specific structured facts
    truncated: bool = False


# ----------------------------------------------------------------------
# Handler base class + registry.
# ----------------------------------------------------------------------
class BaseHandler:
    """Override `_extract_text` (and optionally `_structure`) for a new
    file type. `read()` and the default `search()` are built on top of
    those two and rarely need overriding themselves."""

    extensions: Set[str] = set()
    kind: str = "generic"

    def _extract_text(self, path: str) -> Optional[str]:
        """Return the file's textual content, or None if this type has no
        meaningful text representation (e.g. a raw image)."""
        return None

    def _structure(self, path: str, text: Optional[str]) -> Dict[str, Any]:
        """Type-specific structured facts beyond the raw text preview
        (row counts, JSON keys, image dimensions, page counts, ...)."""
        return {}

    def read(self, path: str, max_chars: int = MAX_PREVIEW_CHARS) -> FileHandlerResult:
        text = self._extract_text(path)
        structure = self._structure(path, text)
        if text is None:
            return FileHandlerResult(kind=self.kind, preview=None, structure=structure)
        truncated = len(text) > max_chars
        return FileHandlerResult(kind=self.kind, preview=text[:max_chars], structure=structure, truncated=truncated)

    def search(self, path: str, query: str, max_chars: int = MAX_PREVIEW_CHARS) -> List[Dict[str, Any]]:
        """Default: line-based substring search over extracted text.
        Handlers with no text (images) should return [] -- the base
        implementation already does, since _extract_text returns None."""
        text = self._extract_text(path)
        if not text:
            return []
        hits = []
        q_lower = query.lower()
        for i, line in enumerate(text.splitlines(), start=1):
            if q_lower in line.lower():
                hits.append({"line": i, "text": line.strip()[:300]})
            if len(hits) >= MAX_SEARCH_RESULTS:
                break
        return hits


class TextHandler(BaseHandler):
    extensions = {".txt", ".md", ".log", ".ini", ".cfg", ".yaml", ".yml",
                  ".py", ".js", ".ts", ".html", ".css", ".sh", ".rst"}
    kind = "text"

    def _extract_text(self, path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.warning("TextHandler failed to read %s: %s", path, e)
            return None

    def _structure(self, path: str, text: Optional[str]) -> Dict[str, Any]:
        if text is None:
            return {}
        return {
            "line_count": text.count("\n") + 1,
            "char_count": len(text),
            "word_count": len(text.split()),
        }

class BinaryHandler(BaseHandler):
    """Fallback for any extension with no registered handler. Metadata
    only -- read()/search() never attempt to decode the bytes as text."""
    extensions: Set[str] = set()
    kind = "binary"


class FileTypeRegistry:
    def __init__(self):
        self._by_ext: Dict[str, BaseHandler] = {}
        self._default = BinaryHandler()

    def register(self, handler: BaseHandler) -> None:
        for ext in handler.extensions:
            self._by_ext[ext.lower()] = handler

    def get(self, path: str) -> BaseHandler:
        ext = os.path.splitext(path)[1].lower()
        return self._by_ext.get(ext, self._default)

    def registered_extensions(self) -> List[str]:
        return sorted(self._by_ext.keys())


REGISTRY = FileTypeRegistry()
REGISTRY.register(TextHandler())


def register_handler(handler: BaseHandler) -> None:
    """Public extension point. Add support for a new file type at runtime
    without touching any dispatcher code:

        class DocxHandler(BaseHandler):
            extensions = {".docx"}
            kind = "docx"
            def _extract_text(self, path):
                ...

        register_handler(DocxHandler())

    Every read_file_tool / search_files_tool / list_directory_tool /
    file_info_tool call below picks up the new type immediately.
    """
    REGISTRY.register(handler)


# ----------------------------------------------------------------------
# Deterministic path/query extraction from a raw chat message.
#
# "Deterministic" here means: same input text -> same extracted path,
# every time. It does NOT mean the regex always gets it right -- natural
# language is messy. When extraction fails, the tool returns a structured
# "couldn't find a path" fact (see _error above) rather than guessing;
# the LLM layer can then ask the user to clarify, but the tool itself
# never makes something up.
# ----------------------------------------------------------------------
_QUOTED_PATH_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_PATH_TOKEN_RE = re.compile(
    r'(?:[A-Za-z]:\\[^\s\'"]+|~?/[^\s\'"]+|\.{1,2}/[^\s\'"]+|[^\s\'"]+\.[A-Za-z0-9]{1,8})'
)
_SEARCH_FOR_RE = re.compile(r'(?:search|find|look)\s+for\s+["\']?(.+?)["\']?\s+in\s+(.+)$', re.IGNORECASE)
_LEADING_COMMAND_RE = re.compile(r'^(search|find|look)\s+(for\s+)?', re.IGNORECASE)


def _extract_path(text: str) -> Optional[str]:
    if not text:
        return None
    m = _QUOTED_PATH_RE.search(text)
    if m:
        return m.group(1) or m.group(2)
    m = _PATH_TOKEN_RE.search(text)
    if m:
        return m.group(0).strip(",.;:!?")
    return None


def _extract_search_query_and_path(text: str):
    m = _SEARCH_FOR_RE.search(text)
    if m:
        query = m.group(1).strip().strip("\"'")
        tail = m.group(2)
        path = _extract_path(tail) or tail.strip()
        return query or None, path
    path = _extract_path(text)
    remainder = text.replace(path, "", 1).strip() if path else text
    query = _LEADING_COMMAND_RE.sub("", remainder).strip().strip("\"'")
    return (query or None), path


# ----------------------------------------------------------------------
# Top-level deterministic operations. These are what ash.py's
# _deterministic_execute calls -- one function per intent, same calling
# convention as calculator_tool/date_time_tool (raw query string in,
# JSON-serializable dict out, never raises).
# ----------------------------------------------------------------------
def file_info_tool(raw_query: str) -> Dict[str, Any]:
    path = _extract_path(raw_query)
    if not path:
        return _error("file_info", raw_query, "I couldn't find a file or folder path in that request.")
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return _error("file_info", path, "No file or folder exists at that path.")
    try:
        st = os.stat(path)
    except Exception as e:
        return _error("file_info", path, f"Couldn't read metadata for that path: {e}")
    is_dir = os.path.isdir(path)
    handler = None if is_dir else REGISTRY.get(path)
    return _ok(
        "file_info", path,
        is_directory=is_dir,
        size_bytes=st.st_size,
        modified=datetime.fromtimestamp(st.st_mtime).isoformat(),
        created=datetime.fromtimestamp(st.st_ctime).isoformat(),
        extension=(None if is_dir else os.path.splitext(path)[1].lower()),
        detected_type=("directory" if is_dir else handler.kind),
    )


def list_directory_tool(raw_query: str) -> Dict[str, Any]:
    path = os.path.expanduser(_extract_path(raw_query) or ".")
    if not os.path.isdir(path):
        return _error("list_directory", path, "That doesn't look like an existing folder.")
    try:
        names = sorted(os.listdir(path))
    except Exception as e:
        return _error("list_directory", path, f"Couldn't list that folder: {e}")

    entries = []
    for name in names:
        full = os.path.join(path, name)
        try:
            st = os.stat(full)
            entries.append({
                "name": name,
                "is_directory": os.path.isdir(full),
                "size_bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
                "extension": os.path.splitext(name)[1].lower(),
            })
        except Exception:
            continue  # skip unreadable entries rather than failing the whole listing

    truncated = len(entries) > MAX_LIST_ENTRIES
    return _ok("list_directory", path, entry_count=len(entries),
               entries=entries[:MAX_LIST_ENTRIES], truncated=truncated)


def read_file_tool(raw_query: str, max_chars: int = MAX_PREVIEW_CHARS) -> Dict[str, Any]:
    path = _extract_path(raw_query)
    if not path:
        return _error("read_file", raw_query, "I couldn't find a file path in that request.")
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return _error("read_file", path, "No file exists at that path.")
    handler = REGISTRY.get(path)
    try:
        result = handler.read(path, max_chars)
    except Exception as e:
        return _error("read_file", path, f"Couldn't read that file: {e}")
    return _ok("read_file", path, file_type=result.kind, preview=result.preview,
               structure=result.structure, truncated=result.truncated)


def search_files_tool(raw_query: str) -> Dict[str, Any]:
    query, root = _extract_search_query_and_path(raw_query)
    if not query:
        print(f"[search_files_tool] no query extracted from: {raw_query} , query: {query}, root: {root}", file=sys.stderr, flush=True)
        return _error("search_files", raw_query, "I couldn't tell what you want to search for.")
    root = os.path.expanduser(root or ".")

    if os.path.isfile(root):
        files = [root]
    elif os.path.isdir(root):
        files = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                files.append(os.path.join(dirpath, fname))
                if len(files) >= MAX_SEARCH_FILES_SCANNED:
                    break
            if len(files) >= MAX_SEARCH_FILES_SCANNED:
                break
    else:
        print(f"[search_files_tool] path does not exist: {root} (raw_query: {raw_query})", file=sys.stderr, flush=True)
        return _error("search_files", root, "That path doesn't exist.")

    matches = []
    files_scanned = 0
    for f in files:
        files_scanned += 1
        handler = REGISTRY.get(f)
        try:
            hits = handler.search(f, query, MAX_PREVIEW_CHARS)
        except Exception as e:
            logger.debug("search failed for %s: %s", f, e)
            continue
        for h in hits:
            matches.append({"file": f, **h})
        if len(matches) >= MAX_SEARCH_RESULTS:
            break

    return _ok("search_files", root, query=query, files_scanned=files_scanned,
               match_count=len(matches), matches=matches[:MAX_SEARCH_RESULTS])


def write_file_tool(path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
    """Not wired to an auto-routed chat intent (yet) -- reliably extracting
    *what content to write* from a single free-text message isn't
    something a deterministic parser can do well, and guessing would
    violate the whole point of this module. Available for direct/
    programmatic use, or once ash.py has a structured way to capture
    "content to save" (e.g. the previous assistant message, or an
    explicit multi-turn flow)."""
    path = os.path.expanduser(path)
    if not path:
        return _error("write_file", path, "No path given.")
    if os.path.exists(path) and not overwrite:
        return _error("write_file", path, "That file already exists -- pass overwrite=True to replace it.")
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return _error("write_file", path, f"Couldn't write that file: {e}")
    return _ok("write_file", path, bytes_written=len(content.encode("utf-8")))


def file_manager_status() -> Dict[str, Any]:
    """Debug/introspection helper, same spirit as ASH.status_info()."""
    return {
        "registered_extensions": REGISTRY.registered_extensions(),
        "max_preview_chars": MAX_PREVIEW_CHARS,
        "max_search_results": MAX_SEARCH_RESULTS,
    }


if __name__ == "__main__":
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="ash_filemgr_test_")
    try:
        with open(os.path.join(tmp, "notes.txt"), "w") as f:
            f.write("Line one\nTODO: fix the parser\nLine three\n")
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump({"debug": True, "retries": 3, "name": "ash"}, f)
        with open(os.path.join(tmp, "data.csv"), "w") as f:
            f.write("name,score\nAda,95\nGrace,98\n")
        with open(os.path.join(tmp, "blob.bin"), "wb") as f:
            f.write(b"\x00\x01\x02\x03")

        print("=== list_directory_tool ===")
        print(json.dumps(list_directory_tool(f'list files in "{tmp}"'), indent=2))

        notes_path = os.path.join(tmp, "notes.txt")
        print("\n=== file_info_tool (notes.txt) ===")
        print(json.dumps(file_info_tool(f'info about "{notes_path}"'), indent=2))

        print("\n=== read_file_tool (notes.txt) ===")
        print(json.dumps(read_file_tool(f'read "{notes_path}"'), indent=2))

        config_path = os.path.join(tmp, "config.json")
        print("\n=== read_file_tool (config.json) ===")
        print(json.dumps(read_file_tool(f'read "{config_path}"'), indent=2))

        csv_path = os.path.join(tmp, "data.csv")
        print("\n=== read_file_tool (data.csv) ===")
        print(json.dumps(read_file_tool(f'read "{csv_path}"'), indent=2))

        blob_path = os.path.join(tmp, "blob.bin")
        print("\n=== file_info_tool (blob.bin, unregistered extension) ===")
        print(json.dumps(file_info_tool(f'info about "{blob_path}"'), indent=2))

        print("\n=== search_files_tool ===")
        print(json.dumps(search_files_tool(f'search for "TODO" in "{tmp}"'), indent=2))

        print("\n=== read_file_tool: missing file ===")
        print(json.dumps(read_file_tool(f'read "{tmp}/nope.txt"'), indent=2))

        print("\n=== read_file_tool: no path in message at all ===")
        print(json.dumps(read_file_tool("can you read that file for me"), indent=2))

        print("\n=== file_manager_status ===")
        print(json.dumps(file_manager_status(), indent=2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)