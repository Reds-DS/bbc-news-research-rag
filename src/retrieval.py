import asyncio
from src.database import ChromaDB
from src.bm25_search import BM25Index, reciprocal_rank_fusion


class Retriever:
    def __init__(self, mode: str = "hybrid"):
        self.mode = mode
        self.db = ChromaDB()
        self._bm25 = None

        if mode in ("keyword", "hybrid"):
            self._bm25 = BM25Index(self.db)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.mode == "semantic":
            return self.db.search(query, limit=limit)

        if self.mode == "keyword":
            return self._bm25.search(query, limit=limit)

        # Hybrid: combine both with RRF
        semantic = self.db.search(query, limit=limit)
        keyword = self._bm25.search(query, limit=limit)
        return reciprocal_rank_fusion(semantic, keyword)[:limit]

    async def asearch(self, query: str, limit: int = 5) -> list[dict]:
        if self.mode == "semantic":
            return await self.db.asearch(query, limit=limit)

        if self.mode == "keyword":
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._bm25.search, query, limit)

        # Hybrid: run both searches in parallel
        semantic_task = self.db.asearch(query, limit=limit)
        loop = asyncio.get_event_loop()
        keyword_task = loop.run_in_executor(None, self._bm25.search, query, limit)

        semantic, keyword = await asyncio.gather(semantic_task, keyword_task)
        return reciprocal_rank_fusion(semantic, keyword)[:limit]
