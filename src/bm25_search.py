from rank_bm25 import BM25Okapi
from src.database import ChromaDB
from src.config import HYBRID_BETA


class BM25Index:
    """In-memory BM25 index built from ChromaDB collection."""

    def __init__(self, db: ChromaDB):
        """Build an in-memory BM25 keyword index from all documents in ChromaDB.

        Fetches every document and its metadata from the ChromaDB collection,
        tokenises the text (lowercased whitespace split), and constructs a
        BM25Okapi index for keyword-based retrieval.

        Args:
            db: An initialised ChromaDB instance whose collection contains the
                documents to index.
        """
        self.db = db
        self._documents = []
        self._corpus = []
        self._bm25 = None
        self._build_index()

    def _build_index(self):
        """Fetch all documents from ChromaDB and build the BM25Okapi index.

        Accesses the underlying ChromaDB collection directly to retrieve raw
        document texts and metadata. Each document is tokenised via lowercase
        whitespace splitting to form the BM25 corpus. Called automatically by
        __init__; should not be called again after construction.
        """
        collection = self.db._store._collection
        batch_size = 500
        offset = 0
        all_documents = []
        all_metadatas = []
        while True:
            results = collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            if not docs:
                break
            all_documents.extend(docs)
            all_metadatas.extend(metas)
            if len(docs) < batch_size:
                break
            offset += batch_size

        # Reuse variable name expected by the rest of the method
        results = {"documents": all_documents, "metadatas": all_metadatas}

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
        """Search the BM25 index and return the top-k matching articles.

        Tokenises the query (lowercase whitespace split) and scores every
        document in the corpus using BM25Okapi. Returns the highest-scoring
        documents sorted by descending BM25 score.

        Args:
            query: The natural-language search query.
            limit: Maximum number of results to return (default 5).

        Returns:
            list[dict]: Ranked list of article dicts, each containing keys:
                'title', 'description', 'pubDate', 'link', 'content', and
                'distance' (BM25 score — higher means more relevant).
        """
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
    """Normalize a list of scores to the [0, 1] range using min-max scaling.

    When all scores are identical (span == 0), returns a list of 0.0 values
    to avoid division by zero.

    Args:
        scores: Raw float scores to normalize.

    Returns:
        list[float]: Normalized scores in [0, 1], preserving relative order.
    """
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

    Each result set is min-max normalized independently, then a weighted
    combination produces the final score:
        final_score = beta * semantic_norm + (1 - beta) * keyword_norm

    Documents appearing in both result sets have their semantic and keyword
    contributions summed. Documents appearing in only one set receive a
    contribution only from that set.

    Args:
        semantic_results: Articles from the vector (embedding) search, each
            with a 'distance' key (cosine distance) and a 'link' key for
            deduplication.
        keyword_results: Articles from the BM25 keyword search, each with
            a 'distance' key (BM25 score) and a 'link' key.
        beta: Weight for semantic scores in [0, 1]. A value of 1.0 means
            semantic-only, 0.0 means keyword-only. Defaults to HYBRID_BETA
            from config.py (0.7).

    Returns:
        list[dict]: Merged and deduplicated articles sorted by descending
            combined score. The 'distance' field is replaced with the
            fused score.
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
