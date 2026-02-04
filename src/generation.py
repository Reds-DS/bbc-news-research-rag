import requests
from langchain_ollama import OllamaLLM
from src.config import OLLAMA_URL, OLLAMA_MODEL


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.llm = OllamaLLM(base_url=self.base_url, model=self.model)

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except requests.RequestException:
            return []

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt)


class RAGGenerator:
    def __init__(self):
        self.client = OllamaClient()

    def _build_prompt(self, query: str, context: list[dict]) -> str:
        context_text = "\n\n---\n\n".join([
            f"Title: {article['title']}\n"
            f"Date: {article['pubDate']}\n"
            f"Content: {article['description']}"
            for article in context
        ])

        prompt = f"""You are a helpful assistant answering questions about BBC News articles.

Based on the following news articles, answer the user's question. If the answer cannot be found in the provided articles, say so.

ARTICLES:
{context_text}

QUESTION: {query}

ANSWER:"""
        return prompt

    def answer(self, query: str, context: list[dict]) -> str:
        prompt = self._build_prompt(query, context)
        return self.client.generate(prompt)
