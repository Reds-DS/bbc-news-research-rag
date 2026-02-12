from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from src.embeddings import get_embeddings


class ChromaDB:
    """Wrapper around LangChain's Chroma vector store for BBC News articles."""

    def __init__(self):
        """Initialize ChromaDB with the configured collection and embedding function.

        Creates a LangChain Chroma vector store backed by persistent storage at
        CHROMA_PERSIST_DIR. Uses cosine distance for HNSW indexing and the
        HuggingFace embedding function from src.embeddings.

        The collection is created automatically if it does not yet exist on disk.
        """
        self._embedding_function = get_embeddings()
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embedding_function,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def collection_exists(self) -> bool:
        """Check whether the ChromaDB collection contains any documents.

        Returns:
            bool: True if the collection has at least one document, False otherwise.
        """
        return self._store._collection.count() > 0

    def delete_collection(self):
        """Delete the existing collection and recreate an empty one.

        Drops all stored documents and their embeddings from persistent storage,
        then immediately recreates the collection so the ChromaDB instance
        remains usable for subsequent add_documents() calls.
        """
        self._store.delete_collection()
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embedding_function,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[Document]):
        """Add a batch of LangChain Documents to the collection.

        Each document's page_content is embedded via the configured HuggingFace
        model and stored alongside its metadata in the ChromaDB collection.

        Args:
            documents: List of LangChain Document objects, each containing
                page_content (text to embed) and metadata (title, description,
                pubDate, guid, link).
        """
        self._store.add_documents(documents)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Run a synchronous cosine-similarity search against the collection.

        Embeds the query string and finds the closest document vectors using
        ChromaDB's HNSW index with cosine distance.

        Args:
            query: The natural-language search query to embed and match.
            limit: Maximum number of results to return (default 5).

        Returns:
            list[dict]: Ranked list of article dicts, each containing keys:
                'title', 'description', 'pubDate', 'link', and 'distance'
                (cosine distance — lower means more similar).
        """
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
        """Return the total number of documents stored in the collection.

        Returns:
            int: Document count in the ChromaDB collection.
        """
        return self._store._collection.count()

    async def asearch(self, query: str, limit: int = 5) -> list[dict]:
        """Asynchronous version of search(), backed by a thread-pool executor.

        LangChain's Chroma asimilarity_search_with_score delegates the blocking
        ChromaDB call to asyncio's default thread-pool executor, making it safe
        to use inside async pipelines without blocking the event loop.

        Args:
            query: The natural-language search query to embed and match.
            limit: Maximum number of results to return (default 5).

        Returns:
            list[dict]: Same structure as search() — ranked article dicts with
                keys 'title', 'description', 'pubDate', 'link', 'distance'.
        """
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
