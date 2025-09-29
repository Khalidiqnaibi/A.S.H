from ....infrastructure.LangChain_Documents_Converter.DocumentConverter import DocumentConverter
from ....infrastructure.LangChain_Documents_Converter.split_text_to_document import split_text_to_docs
from langchain.schema import Document
from typing import List
import csv

class CSVConverter(DocumentConverter):
    def convert(self, source_path: str, *, chunk_size: int = 1000, chunk_overlap: int = 100, base_metadata = [{"source": "csv"}]
) -> List[Document]:
        # Convert CSV rows into a single text blob, then chunk.
        with open(source_path, "r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.reader(fh)
            # join rows with comma and newline; keep rows small enough for chunker to handle
            lines = [", ".join(row) for row in reader]
            text = "\n".join(lines)
        return split_text_to_docs(text, base_metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
