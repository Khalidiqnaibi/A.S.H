from ....infrastructure.LangChain_Documents_Converter.DocumentConverter import DocumentConverter
from ....infrastructure.LangChain_Documents_Converter.split_text_to_document import split_text_to_docs
from langchain.schema import Document
from typing import List
import os

class BaseConverter(DocumentConverter):
    """
    Fallback converter: attempts to read file as text, or if binary returns a single document with metadata.
    """
    def convert(self, source_path: str, *, chunk_size: int = 1000, chunk_overlap: int = 100, base_metadata = [{"source": "generic_text"}]) -> List[Document]:
        try:
            with open(source_path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            return split_text_to_docs(text, base_metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except Exception:
            # binary fallback
            metadata = {"source": source_path, "type": "binary"}
            # Don't attempt to chunk binary content; store a placeholder note
            return [Document(page_content=f"[binary file stored: {os.path.basename(source_path)}]", metadata=metadata)]
