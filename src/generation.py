import requests
from langchain_ollama import OllamaLLM
from src.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE


class OllamaClient:
    """Client for interacting with a local Ollama LLM instance."""

    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        """Initialize the Ollama client with the given base URL and model."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.llm = OllamaLLM(base_url=self.base_url, model=self.model, temperature=OLLAMA_TEMPERATURE)

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """Return a list of model names available on the Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except requests.RequestException:
            return []

    def generate(self, prompt: str) -> str:
        """Generate a response synchronously."""
        return self.llm.invoke(prompt)

    async def agenerate(self, prompt: str) -> str:
        """Generate a response asynchronously."""
        return await self.llm.ainvoke(prompt)


class RAGGenerator:
    """Generate answers to questions using retrieved BBC News articles as context."""

    def __init__(self):
        """Initialize the RAG generator with an Ollama client."""
        self.client = OllamaClient()

    def _build_prompt(self, query: str, context: list[dict]) -> str:
        """Build a RAG prompt from the user query and retrieved articles."""
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
        """Answer a question synchronously using retrieved context."""
        prompt = self._build_prompt(query, context)
        return self.client.generate(prompt)

    async def aanswer(self, query: str, context: list[dict]) -> str:
        """Answer a question asynchronously using retrieved context."""
        prompt = self._build_prompt(query, context)
        return await self.client.agenerate(prompt)
