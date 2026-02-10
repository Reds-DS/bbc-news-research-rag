from rank_bm25 import BM25Okapi
from src.database import ChromaDB
from src.config import HYBRID_BETA


class BM25Index:
    """In-memory BM25 index built from ChromaDB collection."""

    def __init__(self, db: ChromaDB):
        """Build the BM25 index from all documents in the given ChromaDB instance."""
        self.db = db
        self._documents = []
        self._corpus = []
        self._bm25 = None
        self._build_index()

    def _build_index(self):
        """Fetch all documents from ChromaDB and build the BM25Okapi index."""
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
        """Search the BM25 index and return the top-k matching article dicts."""
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


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] using min-max scaling."""
    min_s = min(scores)
    max_s = max(scores)
    span = max_s - min_s
    if span == 0:
        return [0.0] * len(scores)
    return [(s - min_s) / span for s in scores]


def beta_score_fusion(
    semantic_results: list[dict],
    keyword_results: list[dict],
    beta: float | None = None,
) -> list[dict]:
    """Combine semantic and keyword results using beta-weighted score fusion.

    final_score = beta * semantic_norm + (1 - beta) * keyword_norm
    """
    if beta is None:
        beta = HYBRID_BETA

    # Normalize scores per result list
    if semantic_results:
        sem_scores = _min_max_normalize([d["distance"] for d in semantic_results])
    else:
        sem_scores = []
    if keyword_results:
        kw_scores = _min_max_normalize([d["distance"] for d in keyword_results])
    else:
        kw_scores = []

    # Build score map: link -> (combined_score, doc)
    scores: dict[str, tuple[float, dict]] = {}

    for doc, norm in zip(semantic_results, sem_scores):
        link = doc["link"]
        scores[link] = (beta * norm, doc)

    for doc, norm in zip(keyword_results, kw_scores):
        link = doc["link"]
        if link in scores:
            prev_score, prev_doc = scores[link]
            scores[link] = (prev_score + (1 - beta) * norm, prev_doc)
        else:
            scores[link] = ((1 - beta) * norm, doc)

    # Sort descending by combined score
    sorted_results = sorted(scores.values(), key=lambda x: x[0], reverse=True)

    return [
        {**doc, "distance": combined}
        for combined, doc in sorted_results
    ]
