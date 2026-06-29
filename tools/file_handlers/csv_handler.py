from tools import BaseHandler ,FileHandlerResult ,REGISTRY
import logging
from typing import Optional , Dict ,Any 
import csv , io

logger = logging.getLogger("ash.files_handlers")


class CSVHandler(BaseHandler):
    extensions = {".csv", ".tsv"}
    kind = "csv"

    def _delimiter(self, path: str) -> str:
        return "\t" if path.lower().endswith(".tsv") else ","

    def _extract_text(self, path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.warning("CSVHandler failed to read %s: %s", path, e)
            return None

    def _structure(self, path: str, text: Optional[str]) -> Dict[str, Any]:
        if text is None:
            return {}
        try:
            rows = list(csv.reader(io.StringIO(text), delimiter=self._delimiter(path)))
        except Exception as e:
            return {"valid_csv": False, "parse_error": str(e)}
        if not rows:
            return {"valid_csv": True, "row_count": 0, "columns": []}
        header, body = rows[0], rows[1:]
        return {"valid_csv": True, "columns": header, "row_count": len(body), "preview_rows": body[:10]}

    def read(self, path: str, max_chars: int) -> FileHandlerResult:
        structure = self._structure(path, self._extract_text(path))
        cols = structure.get("columns", [])
        rows = structure.get("preview_rows", [])
        lines = ([" | ".join(cols)] if cols else []) + [" | ".join(r) for r in rows]
        preview = "\n".join(lines)
        truncated = bool(preview[max_chars:]) or structure.get("row_count", 0) > len(rows)
        return FileHandlerResult(kind=self.kind, preview=preview[:max_chars], structure=structure, truncated=truncated)
    

REGISTRY.register(CSVHandler)
