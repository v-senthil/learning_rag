# Learning RAG — From Scratch to Production

A hands-on, step-by-step journey building a **Retrieval-Augmented Generation (RAG)** system from the ground up. No frameworks, no magic — just Python, Ollama, and first-principles thinking.

This project walks through every concept needed to build a production-quality RAG pipeline: embeddings, tokenization, chunking, vector search, BM25, hybrid retrieval, evaluation, and a full-stack web UI.

## What You'll Find Here

### 20 Progressive Python Scripts (`src/`)

Each script builds on the last, numbered sequentially so you can follow the learning path:

| # | Script | What It Does |
|---|--------|-------------|
| 01 | `01_first_embedding.py` | Send text to Ollama and inspect the 768-dim vector that comes back |
| 02 | `02_similarity.py` | Cosine similarity and dot product between sentence embeddings |
| 03 | `03_tokenization_peek.py` | See how BPE tokenization breaks text into subword tokens |
| 04 | `04_vocab_proof.py` | Prove that tokens are whole chunks mapped to integer IDs |
| 05 | `05_why_chunk.py` | Empirically show that chunking beats whole-document embedding for retrieval |
| 06 | `06_chunking_strategies.py` | Compare 4 chunking strategies side-by-side on the same query |
| 07 | `07_chunk_all_docs.py` | Chunk all markdown docs and build a JSON vector store (28 chunks) |
| 08 | `08_retrieve.py` | Load the vector store, embed a query, return top-k matches — the "R" in RAG |
| 09 | `09_rebuild_with_prefixes.py` | Rebuild the store with `search_document:` prefix (nomic-embed-text best practice) |
| 10 | `10_retrieve_compare.py` | A/B test: retrieval with vs. without prefixes across 5 queries |
| 11 | `11_rag.py` | End-to-end RAG: retrieve → prompt → generate with citations and refusal |
| 12 | `12_bm25_from_scratch.py` | BM25 keyword search implemented from scratch, validated against `rank-bm25` |
| 13 | `13_hybrid_retrieve.py` | Hybrid retrieval: BM25 + vector cosine, min-max normalized, alpha-blended |
| 14 | `14_eval_runner.py` | Eval runner: test 3 retrieval modes against a 23-query labeled eval set |
| 15 | `15_umap_map.py` | UMAP projection of embedding space → interactive Plotly HTML scatter |
| 16 | `16_streamlit_debugger.py` | Streamlit UI: side-by-side retrieval rankings, LLM answer, UMAP visualization |
| 17 | `17_ingest.py` | Reusable CLI: ingest any PDF/MD/TXT → chunked, embedded JSON store |
| 18 | `18_ask.py` | Reusable CLI: hybrid Q&A over any ingested store (REPL or one-shot) |
| 19 | `19_rag_app.py` | Full Streamlit app: upload, ingest, query, inspect prompts, UMAP map |
| 20 | `20_fastapi_server.py` | FastAPI backend + web UI with streaming, model switching, store management |

**~3,400 lines of Python** — all written from scratch, no LangChain, no LlamaIndex.

### Knowledge Base (`data/`)

Five markdown documents that serve as the retrieval corpus:

- `embeddings_basics.md` — What embeddings are, cosine similarity, vector dimensions
- `transformer_architecture.md` — Self-attention, multi-head attention, positional encoding
- `rag_overview.md` — The RAG pipeline, chunking, retrieval, generation
- `vector_databases.md` — Vector DB concepts, indexing, ANN search
- `python_tips.md` — Python idioms and tips

### Evaluation Set (`eval_set.json`)

23 labeled queries across 5 categories:
- **Direct factoid** — exact answers found in one document
- **Paraphrase** — same question worded differently
- **Rare keyword** — tests keyword-dependent retrieval (BM25 signal)
- **Cross-file synthesis** — answers spanning multiple documents
- **Out-of-scope refusal** — questions the system should refuse to answer

Used by `src/14_eval_runner.py` to benchmark vector-only, BM25-only, and hybrid retrieval.

### Learning Notes (`notes/`)

Seven detailed learning journals documenting the journey:

1. `01_embeddings_and_similarity.md` — Embeddings, cosine similarity, tokenization deep dive
2. `02_chunking.md` — Why chunking matters, strategies, building the vector store
3. `03_retrieval_and_prefixes.md` — Vector search, query prefixes, A/B testing retrieval
4. `04_end_to_end_rag.md` — The full RAG pipeline with citations and refusal
5. `05_bm25_and_hybrid.md` — BM25 from scratch, hybrid blending with alpha tuning
6. `06_evals_and_tuning.md` — Building the eval set, running benchmarks, analyzing results
7. `07_reusable_rag_tool.md` — CLI tools, Streamlit app, FastAPI server

### Web UI (`static/` + FastAPI)

A polished web interface ("RAG Studio") served by the FastAPI backend:
- Upload and ingest documents (PDF, Markdown, plain text)
- Ask questions with streaming answers
- Inspect retrieved sources with similarity scores
- Switch between LLM and embedding models
- Save and load vector stores

## Architecture

```
                    ┌─────────────┐
                    │  Documents  │  PDF / MD / TXT
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Chunking  │  Recursive, 300 chars, 50 overlap
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Embedding  │  Ollama nomic-embed-text (768-dim)
                    └──────┬──────┘      with "search_document:" prefix
                           │
                    ┌──────▼──────┐
                    │ Vector Store│  JSON file (chunks + vectors)
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐ ┌──────▼──────┐ ┌───────▼──────┐
   │   Vector    │ │    BM25     │ │    Hybrid    │
   │   Search    │ │   Search    │ │   (α blend)  │
   │  (cosine)   │ │ (keyword)   │ │              │
   └──────┬──────┘ └──────┬──────┘ └───────┬──────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────▼──────┐
                    │  LLM (RAG)  │  Ollama llama3.2
                    └──────┬──────┘  Grounded answer + citations
                           │
                    ┌──────▼──────┐
                    │   Output    │  Answer with [source] references
                    └─────────────┘  + refusal for out-of-scope queries
```

## Key Concepts Covered

- **Embeddings** — How text becomes a 768-dimensional vector via tokenization → embedding matrix → transformer layers → attention → pooling → normalization
- **Cosine Similarity** — Measuring semantic closeness between vectors
- **Tokenization** — BPE subword tokenization, vocabulary lookup, token-to-integer mapping
- **Chunking** — Why it matters, recursive character splitting, overlap for context preservation
- **Query Prefixes** — nomic-embed-text's asymmetric `search_document:` / `search_query:` prefixes
- **BM25** — Term frequency, inverse document frequency, length normalization — no ML needed
- **Hybrid Retrieval** — Combining BM25 keyword signal with vector semantic signal via alpha blending
- **Evaluation** — Labeled eval sets, top-1/top-3 accuracy, per-category analysis
- **RAG Pipeline** — Retrieve → augment prompt → generate with citations → refuse when unsupported
- **UMAP Visualization** — Projecting 768-dim embeddings to 2D to see cluster structure

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.ai)** installed and running
- Models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.2
  ```

## Setup

```bash
git clone https://github.com/v-senthil/learning_rag.git
cd learning_rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Follow the Learning Path

Run scripts in order to build understanding step by step:

```bash
python src/01_first_embedding.py    # Your first embedding
python src/02_similarity.py          # Cosine similarity
# ... continue through 01-14 for the core concepts
```

### Run the Eval Suite

```bash
python src/14_eval_runner.py
```

Benchmarks vector-only, BM25-only, and hybrid retrieval against 23 labeled queries.

### Use the CLI Tools

```bash
# Ingest any document
python src/17_ingest.py path/to/document.pdf --output my_store.json

# Ask questions (interactive REPL)
python src/18_ask.py --store my_store.json

# One-shot query
python src/18_ask.py --store my_store.json --query "What is self-attention?"
```

### Launch the Streamlit App

```bash
streamlit run src/19_rag_app.py
```

Full-featured UI: upload documents, ask questions, inspect retrieval, visualize embeddings.

### Launch the FastAPI Server

```bash
python src/20_fastapi_server.py
```

REST API + web UI at `http://localhost:8000`. Swagger docs at `/docs`.

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + store size |
| `GET` | `/models` | List available LLM and embedding models |
| `POST` | `/models/llm` | Switch LLM model |
| `POST` | `/models/embed` | Switch embedding model |
| `GET` | `/stores` | List saved vector stores |
| `POST` | `/stores/{name}/load` | Load a saved store |
| `POST` | `/stores/{name}/save` | Save current store |
| `DELETE` | `/store` | Clear in-memory store |
| `POST` | `/ingest` | Upload and ingest documents |
| `POST` | `/ask` | Query with JSON, get answer + sources |
| `POST` | `/ask/stream` | Streaming NDJSON response |

## Stack

| Component | Tool |
|-----------|------|
| Embeddings | [Ollama](https://ollama.ai) + nomic-embed-text |
| LLM | Ollama + llama3.2 |
| Vector math | NumPy, scikit-learn |
| Keyword search | BM25 (from scratch + rank-bm25) |
| Visualization | UMAP, Plotly, Matplotlib |
| Interactive UI | Streamlit |
| API server | FastAPI + Uvicorn |
| PDF parsing | pypdf |

## Project Structure

```
learning_rag/
├── data/                  # Knowledge base (5 markdown documents)
│   ├── embeddings_basics.md
│   ├── transformer_architecture.md
│   ├── rag_overview.md
│   ├── vector_databases.md
│   └── python_tips.md
├── src/                   # 20 progressive Python scripts
│   ├── 01_first_embedding.py  ... 20_fastapi_server.py
├── notes/                 # 7 learning journal entries
│   ├── 01_embeddings_and_similarity.md  ... 07_reusable_rag_tool.md
├── static/                # Web UI for FastAPI server
│   ├── index.html
│   ├── style.css
│   └── app.js
├── stores/                # Saved vector stores (JSON)
├── eval_set.json          # 23 labeled evaluation queries
├── vector_store.json      # Original vector store (no prefixes)
├── vector_store_v2.json   # Improved store (with search_document: prefix)
├── requirements.txt
└── README.md
```

## Learning Path

This project is part of a staged AI learning progression:

- **Stage 1** — Python for AI, ML vocabulary, Pydantic, async, FastAPI basics
- **Stage 2** — Embeddings → vector search → RAG (this project)
- **Stage 3** — Agents, MCP, observability, deployment
- **Stage 4** — Local LLMs, quantization, fine-tuning

## License

This is a personal learning project. Feel free to use it as a reference for your own RAG learning journey.
