from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create a HuggingFaceEmbeddings instance with normalized embeddings."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
