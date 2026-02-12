import asyncio

from sentence_transformers import CrossEncoder

MODEL_ALIASES = {
    "heavy": "BAAI/bge-reranker-v2-m3",
    "light": "cross-encoder/ms-marco-MiniLM-L12-v2",
}


class Reranker:
    """Cross-encoder reranker that rescores retrieved documents against a query."""

    def __init__(self, model_name: str):
        """Initialize the reranker with a cross-encoder model.

        Resolves short aliases ('light', 'heavy') to full HuggingFace model
        IDs via MODEL_ALIASES. If the name is not an alias, it is treated as
        a full model ID and loaded directly.

        Args:
            model_name: Either an alias ('light' for ms-marco-MiniLM-L12-v2,
                'heavy' for bge-reranker-v2-m3) or a full HuggingFace model
                identifier string.
        """
        full_name = MODEL_ALIASES.get(model_name, model_name)
        self.model = CrossEncoder(full_name)

    def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        """Rescore documents using the cross-encoder and return the top_k results.

        Builds (query, "title description") text pairs and feeds them through
        the CrossEncoder.predict() method. Each document's 'distance' field is
        replaced with the cross-encoder relevance score, and results are sorted
        in descending order (higher = more relevant).

        Args:
            query: The original search query used for scoring relevance.
            documents: List of article dicts from the initial retrieval stage.
                Each must contain 'title' and 'description' keys.
            top_k: Number of top-scoring documents to return after reranking.

        Returns:
            list[dict]: The top_k documents sorted by descending cross-encoder
                score, with the 'distance' field set to the reranker score.
        """
        if not documents:
            return documents

        pairs = [
            (query, f"{doc['title']} {doc['description']}")
            for doc in documents
        ]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["distance"] = float(score)

        ranked = sorted(documents, key=lambda d: d["distance"], reverse=True)
        return ranked[:top_k]

    async def arerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        """Async wrapper around rerank(), offloading CPU-bound inference to a thread pool.

        Cross-encoder scoring is computationally expensive and would block the
        event loop, so this method delegates to asyncio's default thread-pool
        executor.

        Args:
            query: The original search query used for scoring relevance.
            documents: List of article dicts from the initial retrieval stage.
            top_k: Number of top-scoring documents to return after reranking.

        Returns:
            list[dict]: Same as rerank() — top_k documents sorted by descending
                cross-encoder score.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, documents, top_k)
