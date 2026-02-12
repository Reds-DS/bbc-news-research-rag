import streamlit as st
from src.retrieval import Retriever
from src.generation import RAGGenerator

# Page config
st.set_page_config(page_title="BBC News Q&A", page_icon="📰")
st.title("📰 BBC News Q&A")
st.caption("Ask questions about BBC News articles")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Initialize components (cached)
@st.cache_resource
def get_retriever():
    """Create and cache a hybrid Retriever instance for the Streamlit session.

    Uses Streamlit's @cache_resource decorator so the Retriever (including
    its ChromaDB connection and BM25 index) is initialised only once and
    shared across all user interactions in the same server process.

    Returns:
        Retriever: A Retriever configured with hybrid search mode.
    """
    return Retriever(mode="hybrid")


@st.cache_resource
def get_generator():
    """Create and cache a RAGGenerator instance for the Streamlit session.

    Uses Streamlit's @cache_resource decorator so the RAGGenerator (and its
    underlying OllamaClient connection) is initialised only once and shared
    across all user interactions in the same server process.

    Returns:
        RAGGenerator: A generator configured with the default Ollama model.
    """
    return RAGGenerator()


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.write(f"- {src}")

# Chat input
if prompt := st.chat_input("Ask a question about BBC News..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching articles..."):
            retriever = get_retriever()
            context = retriever.search(prompt, limit=5)

        with st.spinner("Generating answer..."):
            generator = get_generator()
            answer = generator.answer(prompt, context)

        st.markdown(answer)

        # Show sources
        sources = [f"{a['title']} ({a['pubDate']})" for a in context]
        with st.expander("Sources"):
            for src in sources:
                st.write(f"- {src}")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
