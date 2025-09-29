# from typing import List
# from domain.interfaces.IEmbedder import IEmbedder
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from ...infrastructure.LangChain_Documents_Converter.LangChainToDocumentConverter import LangChainToDocument

# class HuggingFaceEmbedder(IEmbedder):
#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#         self._model = HuggingFaceEmbeddings(model_name=model_name)

#     def embed_text(self, text: str) -> List[float]:
#         return self._model.embed_query(text)

#     def embed_documents(self, path: str, *, chunk_size: int = 1000, chunk_overlap: int = 100 ):
#         embed = LangChainToDocument.from_path(
#             path = path,
#             chunk_size = chunk_size,
#             chunk_overlap = chunk_overlap
#         )

#         return self._model.embed_documents(embed)


#Deprecated