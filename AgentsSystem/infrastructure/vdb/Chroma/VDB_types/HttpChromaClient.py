import chromadb
from typing import Dict, Optional
from chromadb.config import Settings

class HttpChromaClient:
    def __init__(
                self,
                headers: Dict[str, str],
                settings: Settings = Settings(),
                host: str = "localhost",
                port: int = 4000, 
                ssl: bool = False,
                tanent: str = "",
                database: str = "",
                ):
        self._vdb = chromadb.HttpClient(
            host = host,
            port = port,
            ssl = ssl,
            headers=headers,
            # tenant=tanent,
            # database=database,
            settings=settings
        )

    def get_vdb_client(self):
        return self._vdb