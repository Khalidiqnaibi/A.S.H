from tools import BaseHandler ,FileHandlerResult ,REGISTRY
import logging
from typing import Optional , Dict ,Any 
from PyPDF2 import PdfReader

logger = logging.getLogger("ash.files_handlers")

class PDFHandler(BaseHandler):
    extensions = {".pdf"}
    kind = "pdf"

    def _extract_text(self, path: str) -> Optional[str]:
        try:
            reader = PdfReader(path)
            self._page_count = len(reader.pages)
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            logger.warning("PDFHandler failed to read %s: %s", path, e)
            return None

    def _structure(self, path: str, text: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if hasattr(self, "_page_count"):
            info["page_count"] = self._page_count
        return info

REGISTRY.register(PDFHandler)
