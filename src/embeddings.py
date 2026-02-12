from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create a HuggingFaceEmbeddings instance configured for this project.

    Loads the embedding model specified by EMBEDDING_MODEL in config.py
    (default: 'Lajavaness/bilingual-embedding-base') with L2-normalized
    output vectors, which is required for cosine similarity in ChromaDB.

    Returns:
        HuggingFaceEmbeddings: A LangChain-compatible embedding function that
            can be passed directly to Chroma as the embedding_function parameter.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
