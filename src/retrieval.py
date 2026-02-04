from src.database import ChromaDB


class Retriever:
    def __init__(self):
        self.db = ChromaDB()

    def search(self, query: str, limit: int = 5) -> list[dict]:
        results = self.db.search(query, limit=limit)
        return results
