from typing import List, Dict
from langchain.schema import Document

class DocumentConverter:
    def convert(self, source_path: str, *, chunk_size: int = 1000, chunk_overlap: int = 100, base_metadata: Dict[str, str]) -> List[Document]:
        raise NotImplementedError()
