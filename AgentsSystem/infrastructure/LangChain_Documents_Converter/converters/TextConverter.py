from ....infrastructure.LangChain_Documents_Converter.DocumentConverter import DocumentConverter
from ....infrastructure.LangChain_Documents_Converter.split_text_to_document import split_text_to_docs
from typing import List
from langchain.schema import Document

class TextConverter(DocumentConverter):
    def convert(self, source_path: str, *, chunk_size: int = 1000, chunk_overlap: int = 100, base_metadata = [{"source": "text"}]
) -> List[Document]:
        with open(source_path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        return split_text_to_docs(text, base_metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

