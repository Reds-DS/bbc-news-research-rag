import requests
from langchain_ollama import OllamaLLM
from src.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE


class OllamaClient:
    """Client for interacting with a local Ollama LLM instance."""

    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        """Initialize the Ollama client with a LangChain OllamaLLM backend.

        Creates a LangChain OllamaLLM instance configured with the specified
        server URL, model name, and temperature (from config.py). The
        temperature is set to 0 by default for deterministic outputs.

        Args:
            base_url: URL of the Ollama server (e.g. 'http://ollama:11434').
                Defaults to OLLAMA_URL from config.py.
            model: Name of the Ollama model to use (e.g. 'llama3.2').
                Defaults to OLLAMA_MODEL from config.py.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.llm = OllamaLLM(base_url=self.base_url, model=self.model, temperature=OLLAMA_TEMPERATURE)

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable by pinging the /api/tags endpoint.

        Returns:
            bool: True if the server responds with HTTP 200 within 5 seconds,
                False on any connection error or non-200 status.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """Fetch the list of locally available model names from the Ollama server.

        Queries the /api/tags endpoint and extracts model names from the
        response. Returns an empty list if the server is unreachable or
        returns an error.

        Returns:
            list[str]: Model name strings (e.g. ['llama3.2', 'mistral']),
                or an empty list on failure.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except requests.RequestException:
            return []

    def generate(self, prompt: str) -> str:
        """Generate a text completion synchronously using the Ollama model.

        Args:
            prompt: The full prompt string to send to the LLM.

        Returns:
            str: The model's generated text response.
        """
        return self.llm.invoke(prompt)

    async def agenerate(self, prompt: str) -> str:
        """Generate a text completion asynchronously using the Ollama model.

        Uses langchain-ollama's native async support (ainvoke), which makes
        a non-blocking HTTP call to the Ollama server.

        Args:
            prompt: The full prompt string to send to the LLM.

        Returns:
            str: The model's generated text response.
        """
        return await self.llm.ainvoke(prompt)


class RAGGenerator:
    """Generate answers to questions using retrieved BBC News articles as context."""

    def __init__(self):
        """Initialize the RAG generator with a default OllamaClient.

        The underlying OllamaClient uses the OLLAMA_URL and OLLAMA_MODEL
        settings from config.py.
        """
        self.client = OllamaClient()

    def _build_prompt(self, query: str, context: list[dict]) -> str:
        """Build the full RAG prompt by combining retrieved articles with the user query.

        Formats each article as "Title / Date / Content" separated by '---'
        dividers, then wraps them in an instruction template that tells the
        LLM to answer using only the provided articles.

        Args:
            query: The user's natural-language question.
            context: List of article dicts from the retriever, each containing
                'title', 'pubDate', and 'description' keys.

        Returns:
            str: The assembled prompt string ready for LLM generation.
        """
        context_text = "\n\n---\n\n".join([
            f"Title: {article['title']}\n"
            f"Date: {article['pubDate']}\n"
            f"Content: {article['description']}"
            for article in context
        ])

        prompt = f"""You are a helpful assistant answering questions based on BBC News articles.

Answer the question directly and concisely using only the information from the articles below. Synthesize relevant details into a clear answer. If the articles contain no relevant information, reply with a single sentence stating that.

ARTICLES:
{context_text}

QUESTION: {query}

ANSWER:"""
        return prompt

    def answer(self, query: str, context: list[dict]) -> str:
        """Answer a question synchronously using retrieved article context.

        Builds a RAG prompt from the query and context, then sends it to
        the Ollama model for generation.

        Args:
            query: The user's natural-language question.
            context: List of article dicts from the retriever.

        Returns:
            str: The LLM-generated answer grounded in the provided articles.
        """
        prompt = self._build_prompt(query, context)
        return self.client.generate(prompt)

    def trace_answer(self, query: str, context: list[dict]) -> tuple[str, str]:
        """Return (prompt, answer) — the full LLM prompt and generated answer."""
        prompt = self._build_prompt(query, context)
        answer = self.client.generate(prompt)
        return prompt, answer

    async def aanswer(self, query: str, context: list[dict]) -> str:
        """Answer a question asynchronously using retrieved article context.

        Async counterpart of answer(). Uses the OllamaClient's native async
        generation to avoid blocking the event loop.

        Args:
            query: The user's natural-language question.
            context: List of article dicts from the retriever.

        Returns:
            str: The LLM-generated answer grounded in the provided articles.
        """
        prompt = self._build_prompt(query, context)
        return await self.client.agenerate(prompt)
