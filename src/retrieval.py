import asyncio
import copy
from src.database import ChromaDB
from src.bm25_search import BM25Index, beta_score_fusion
from src.config import RERANK_FETCH_MULTIPLIER


class Retriever:
    """Unified retriever supporting semantic, keyword, and hybrid search modes."""

    def __init__(self, mode: str = "hybrid", reranker=None, beta: float | None = None):
        """Initialize the retriever with a search mode and optional cross-encoder reranker.

        Creates the underlying ChromaDB instance and, when the mode requires
        keyword matching, also builds an in-memory BM25 index from the full
        collection.

        Args:
            mode: Search strategy — one of 'semantic' (vector only),
                'keyword' (BM25 only), or 'hybrid' (beta-weighted fusion
                of both). Defaults to 'hybrid'.
            reranker: An optional Reranker instance. When provided, the
                retriever over-fetches by RERANK_FETCH_MULTIPLIER and then
                rescores results with the cross-encoder before returning the
                final top-k. Defaults to None (no reranking).
            beta: Hybrid search weight (0.0=keyword only, 1.0=semantic only).
                None uses the HYBRID_BETA value from config.
        """
        self.mode = mode
        self.db = ChromaDB()
        self._bm25 = None
        self.reranker = reranker
        self.beta = beta

        if mode in ("keyword", "hybrid"):
            self._bm25 = BM25Index(self.db)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search for articles synchronously using the configured mode.

        When a reranker is attached, the initial retrieval fetches
        limit * RERANK_FETCH_MULTIPLIER documents, which are then rescored
        by the cross-encoder and trimmed to the requested limit.

        Args:
            query: The natural-language search query.
            limit: Number of final results to return (default 5).

        Returns:
            list[dict]: Ranked article dicts with keys 'title', 'description',
                'pubDate', 'link', and 'distance'.
        """
        fetch_limit = limit * RERANK_FETCH_MULTIPLIER if self.reranker else limit

        if self.mode == "semantic":
            results = self.db.search(query, limit=fetch_limit)
        elif self.mode == "keyword":
            results = self._bm25.search(query, limit=fetch_limit)
        else:
            # Hybrid: beta-weighted score fusion
            semantic = self.db.search(query, limit=fetch_limit)
            keyword = self._bm25.search(query, limit=fetch_limit)
            results = beta_score_fusion(semantic, keyword, beta=self.beta)[:fetch_limit]

        if self.reranker:
            results = self.reranker.rerank(query, results, top_k=limit)

        return results

    def trace_search(self, query: str, limit: int = 5, fetch_limit: int | None = None) -> tuple[list[dict], list[dict]]:
        """Return (pre_rerank_results, post_rerank_results)."""
        if fetch_limit is None:
            fetch_limit = limit * RERANK_FETCH_MULTIPLIER if self.reranker else limit

        if self.mode == "semantic":
            results = self.db.search(query, limit=fetch_limit)
        elif self.mode == "keyword":
            results = self._bm25.search(query, limit=fetch_limit)
        else:
            semantic = self.db.search(query, limit=fetch_limit)
            keyword = self._bm25.search(query, limit=fetch_limit)
            results = beta_score_fusion(semantic, keyword, beta=self.beta)[:fetch_limit]

        if self.reranker:
            pre_rerank = copy.deepcopy(results)
            post_rerank = self.reranker.rerank(query, results, top_k=limit)
            return pre_rerank, post_rerank

        return results, results

    async def asearch(self, query: str, limit: int = 5) -> list[dict]:
        """Async counterpart of search().

        Runs the retrieval stage asynchronously (ChromaDB via thread pool,
        BM25 via executor). If a reranker is attached, the cross-encoder
        rescoring is also offloaded to a thread-pool executor via arerank().

        Args:
            query: The natural-language search query.
            limit: Number of final results to return (default 5).

        Returns:
            list[dict]: Ranked article dicts with keys 'title', 'description',
                'pubDate', 'link', and 'distance'.
        """
        fetch_limit = limit * RERANK_FETCH_MULTIPLIER if self.reranker else limit

        if self.mode == "semantic":
            results = await self.db.asearch(query, limit=fetch_limit)
        elif self.mode == "keyword":
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._bm25.search, query, fetch_limit)
        else:
            # Hybrid: beta-weighted score fusion (parallel)
            semantic_task = self.db.asearch(query, limit=fetch_limit)
            loop = asyncio.get_event_loop()
            keyword_task = loop.run_in_executor(None, self._bm25.search, query, fetch_limit)

            semantic, keyword = await asyncio.gather(semantic_task, keyword_task)
            results = beta_score_fusion(semantic, keyword, beta=self.beta)[:fetch_limit]

        if self.reranker:
            results = await self.reranker.arerank(query, results, top_k=limit)

        return results
