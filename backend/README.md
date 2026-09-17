# PDF Q&A — Local RAG System

Ask questions about any PDF and get accurate, context-grounded answers — built from scratch to understand how Retrieval-Augmented Generation actually works under the hood, not just wrapping a library.

## How it works

1. **Parse** — extract text from an uploaded PDF
2. **Chunk** — split the text into manageable pieces
3. **Embed** — generate vector embeddings locally using `sentence-transformers` (`all-MiniLM-L6-v2`) — no API calls, no cost
4. **Store** — save chunks + embeddings in a local **ChromaDB** vector database
5. **Retrieve** — given a question, embed it and run a similarity search against stored chunks (cosine similarity under the hood) to find the most relevant context
6. **Generate** — feed the retrieved context + question into an LLM (**Groq**, `openai/gpt-oss-120b`) to produce a final natural-language answer

## Tech stack

- **Python**
- `sentence-transformers` — local embedding generation
- `chromadb` — local vector storage & similarity search
- `groq` — LLM inference for answer generation

## Setup

```bash
git clone <your-repo-url>
cd <repo-name>
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

## Usage

```bash
python main.py
```

This parses the PDF, chunks it, generates embeddings, stores them in Chroma, and answers a sample query end to end.

## Example

**Question:** *"What is the shape of the earth?"*

**Answer:** *The Earth is not a perfect sphere. It is an oblate spheroid — roughly spherical but slightly flattened at the poles and bulging at the equator. Its average radius is about 6,371 km.*

## Status / Roadmap

- [x] PDF parsing + chunking
- [x] Local embedding generation
- [x] Chroma vector storage & retrieval
- [x] LLM-based answer generation (Groq)
- [ ] FastAPI backend (PDF upload + question endpoints)
- [ ] Per-user/per-upload data isolation
- [ ] Frontend

## Why this project

Built as a hands-on learning project to understand RAG systems at a first-principles level — tokenization, embedding matrices, pooling, cosine similarity, vector DB internals, and prompt construction — rather than just calling a high-level RAG framework.
