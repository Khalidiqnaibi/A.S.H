from tools import BaseHandler ,FileHandlerResult ,REGISTRY
import logging
from typing import Optional , Dict ,Any 
import json

logger = logging.getLogger("ash.files_handlers")


class JSONHandler(BaseHandler):
    extensions = {".json"}
    kind = "json"

    def _extract_text(self, path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.warning("JSONHandler failed to read %s: %s", path, e)
            return None

    def _structure(self, path: str, text: Optional[str]) -> Dict[str, Any]:
        if text is None:
            return {}
        try:
            data = json.loads(text)
        except Exception as e:
            return {"valid_json": False, "parse_error": str(e)}
        info: Dict[str, Any] = {"valid_json": True}
        if isinstance(data, dict):
            info.update(top_level_type="object", keys=list(data.keys())[:50], key_count=len(data))
        elif isinstance(data, list):
            info.update(top_level_type="array", item_count=len(data))
        else:
            info["top_level_type"] = type(data).__name__
        return info

    def read(self, path: str, max_chars: int) -> FileHandlerResult:
        text = self._extract_text(path)
        structure = self._structure(path, text)
        if text is None:
            return FileHandlerResult(kind=self.kind, preview=None, structure=structure)
        pretty = text
        if structure.get("valid_json"):
            try:
                pretty = json.dumps(json.loads(text), indent=2)
            except Exception:
                pretty = text
        truncated = len(pretty) > max_chars
        return FileHandlerResult(kind=self.kind, preview=pretty[:max_chars], structure=structure, truncated=truncated)
    

REGISTRY.register(JSONHandler())
