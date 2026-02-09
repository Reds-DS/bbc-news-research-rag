from rank_bm25 import BM25Okapi
from src.database import ChromaDB
from src.config import RRF_K


class BM25Index:
    """In-memory BM25 index built from ChromaDB collection."""

    def __init__(self, db: ChromaDB):
        self.db = db
        self._documents = []
        self._corpus = []
        self._bm25 = None
        self._build_index()

    def _build_index(self):
        # Fetch all documents from ChromaDB
        collection = self.db._store._collection
        results = collection.get(include=["documents", "metadatas"])

        for doc, metadata in zip(results["documents"], results["metadatas"]):
            self._documents.append({
                "content": doc,
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "pubDate": metadata.get("pubDate", ""),
                "link": metadata.get("link", ""),
            })
            # Tokenize: simple whitespace split, lowercase
            self._corpus.append(doc.lower().split())

        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if not self._bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]

        results = []
        for idx in top_indices:
            doc = self._documents[idx].copy()
            doc["distance"] = float(scores[idx])  # BM25 score (higher = better)
            results.append(doc)

        return results


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    keyword_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Combine semantic and keyword results using RRF."""
    k = RRF_K

    # Build score map: link -> (rrf_score, doc)
    scores = {}

    for rank, doc in enumerate(semantic_results):
        link = doc["link"]
        rrf = 1.0 / (k + rank + 1)
        if link in scores:
            scores[link] = (scores[link][0] + rrf, doc)
        else:
            scores[link] = (rrf, doc)

    for rank, doc in enumerate(keyword_results):
        link = doc["link"]
        rrf = 1.0 / (k + rank + 1)
        if link in scores:
            scores[link] = (scores[link][0] + rrf, scores[link][1])
        else:
            scores[link] = (rrf, doc)

    # Sort by RRF score descending
    sorted_results = sorted(scores.values(), key=lambda x: x[0], reverse=True)

    # Return docs with RRF score as distance
    return [
        {**doc, "distance": rrf_score}
        for rrf_score, doc in sorted_results
    ]
