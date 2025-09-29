import os
from .LoaderFactory import LoaderFactory
from .split_text_to_document import split_text_to_docs
from typing import List, Dict
from langchain.schema import Document

class LangChainToDocument:
    """
    Facade for converting file paths or raw text into a list of LangChain Documents.
    Use ToDocument.from_path(path) for files or ToDocument.from_text(text, metadata).
    """

    @staticmethod
    def from_path(path: str, *, chunk_size: int = 1000, chunk_overlap: int = 100, base_metadata: Dict[str, str]) -> List[Document]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        _, ext = os.path.splitext(path)
        converter = LoaderFactory.get_converter(os.path.basename(path).split(".")[-1])
        docs = converter.convert(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, base_metadata=base_metadata)
        # Ensure each doc has chunk_id metadata for traceability
        for i, d in enumerate(docs):
            md = dict(d.metadata or {})
            if "chunk_id" not in md:
                md["chunk_id"] = f"{os.path.basename(path)}::{i}"
            d.metadata = md
        return docs

    @staticmethod
    def from_text(text: str, *, chunk_size: int = 1000, chunk_overlap: int = 100, base_metadata = {"source": "user", "type": "text"}) -> List[Document]:
        docs = split_text_to_docs(text, base_metadata=base_metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap,)
        for i, d in enumerate(docs):
            md = dict(d.metadata or {})
            md.setdefault("chunk_id", f"{base_metadata['source']}::{i}")
            d.metadata = md
        return docs
