from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from src.embeddings import get_embeddings


class ChromaDB:
    def __init__(self):
        self._embedding_function = get_embeddings()
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embedding_function,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def collection_exists(self) -> bool:
        return self._store._collection.count() > 0

    def delete_collection(self):
        self._store.delete_collection()
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embedding_function,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[Document]):
        self._store.add_documents(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        results = self._store.similarity_search_with_score(query, k=limit)

        articles = []
        for doc, score in results:
            articles.append({
                "title": doc.metadata.get("title", ""),
                "description": doc.metadata.get("description", ""),
                "pubDate": doc.metadata.get("pubDate", ""),
                "link": doc.metadata.get("link", ""),
                "distance": score,
            })
        return articles

    def get_count(self) -> int:
        return self._store._collection.count()
