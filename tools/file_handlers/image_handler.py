from tools import BaseHandler ,FileHandlerResult ,REGISTRY
import logging
from typing import Optional , Dict ,Any ,List

logger = logging.getLogger("ash.files_handlers")


class ImageHandler(BaseHandler):
    """Metadata only -- there's no useful "text preview" of an image, and
    we are not in the business of running vision models inside a
    deterministic tool layer."""
    extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    kind = "image"

    def _structure(self, path: str, text: Optional[str]) -> Dict[str, Any]:
        try:
            from PIL import Image  # optional dependency
        except Exception as e:
            return {"dimensions_available": False, "reason": f"Pillow not installed ({e})"}
        try:
            with Image.open(path) as img:
                return {"dimensions_available": True, "width": img.width, "height": img.height,
                        "format": img.format, "mode": img.mode}
        except Exception as e:
            return {"dimensions_available": False, "reason": str(e)}

    def search(self, path: str, query: str, max_chars: int) -> List[Dict[str, Any]]:
        return []


REGISTRY.register(ImageHandler())
