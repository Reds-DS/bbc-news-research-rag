import os
from dotenv import load_dotenv

load_dotenv()

# Chroma settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_data")

# Ollama settings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TEMPERATURE = 0

# Collection settings
COLLECTION_NAME = "BBC_NEWS"
EMBEDDING_MODEL = "Lajavaness/bilingual-embedding-base"

# Ingestion settings
BATCH_SIZE = 100
DATA_PATH = "Data/bbc_news.csv"

# Evaluation settings
EVAL_QUESTIONS_PATH = "eval_questions.json"

# OpenAI settings (for evaluation)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EVAL_LLM_MODEL = "gpt-4.1-2025-04-14"
EVAL_EMBEDDING_MODEL = "text-embedding-3-small"

# Hybrid search settings
HYBRID_BETA = 0.7  # 0.0 = keyword only, 1.0 = semantic only

# Reranking settings
RERANK_FETCH_MULTIPLIER = 3