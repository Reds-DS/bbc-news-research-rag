from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from src.embeddings import get_embeddings


class ChromaDB:
    """Wrapper around LangChain's Chroma vector store for BBC News articles."""

    def __init__(self):
        """Initialize ChromaDB with the configured collection and embedding function."""
        self._embedding_function = get_embeddings()
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embedding_function,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def collection_exists(self) -> bool:
        """Return True if the collection has at least one document."""
        return self._store._collection.count() > 0

    def delete_collection(self):
        """Delete and recreate the collection."""
        self._store.delete_collection()
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embedding_function,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[Document]):
        """Add a batch of LangChain Documents to the collection."""
        self._store.add_documents(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Run a synchronous semantic similarity search and return article dicts."""
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
        """Return the number of documents in the collection."""
        return self._store._collection.count()

    async def asearch(self, query: str, limit: int = 5) -> list[dict]:
        """Async search using thread pool executor."""
        results = await self._store.asimilarity_search_with_score(query, k=limit)

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
