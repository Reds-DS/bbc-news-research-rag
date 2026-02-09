# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) system for BBC News articles using:
- **Chroma** - Vector database for semantic search (embedded mode, persistent to disk)
- **LangChain** - Embeddings (`HuggingFaceEmbeddings` with all-MiniLM-L6-v2) and LLM wrapper (`Ollama`)
- **Ollama** - Local LLM inference (llama3.2)

## Commands

### Start Services
```bash
docker-compose up --build -d
```

### Pull Ollama Model (first time)
```bash
docker-compose exec ollama ollama pull llama3.2
```

### CLI Commands
```bash
# Check status
docker-compose exec app python cli.py status

# Ingest data (first time or to recreate)
docker-compose exec app python cli.py ingest --recreate

# Semantic search
docker-compose exec app python cli.py search "Ukraine conflict"

# Ask a question (RAG)
docker-compose exec app python cli.py ask "What happened in Ukraine?"
```

## Architecture

```
src/
├── config.py      # Settings (Chroma persist dir, Ollama URL, model names)
├── embeddings.py  # LangChain HuggingFaceEmbeddings factory
├── database.py    # Chroma vector store wrapper
├── ingestion.py   # CSV loading and indexing pipeline
├── retrieval.py   # Semantic search
└── generation.py  # LangChain Ollama client and RAG prompt
cli.py             # Typer CLI interface
chat.py            # Streamlit chat UI
```

**Data Flow:**
- Ingestion: CSV → Chroma (embedding handled internally by LangChain)
- Query: User question → Chroma semantic search → Ollama generate → Answer

**Chroma Collection:** `BBC_NEWS` with metadata: title, pubDate, guid, link, description

**Docker Services:**
- `ollama` - Port 11434
- `app` - Python application container (Chroma runs embedded inside the app)

