# Retrieval-Augmented Generation (RAG)

## The Core Idea

Large Language Models (LLMs) are powerful but have two big limitations:
1. Their knowledge is frozen at training time.
2. They can hallucinate facts confidently.

RAG solves both problems by giving the LLM **relevant context at query time**,
fetched from your own trusted data. The LLM then answers based on that context
rather than relying purely on memory.

## The RAG Pipeline

A typical RAG system has these stages:

1. **Ingestion**: Load documents, split them into chunks.
2. **Embedding**: Convert each chunk into a vector using an embedding model.
3. **Indexing**: Store vectors in a vector database (FAISS, Chroma, etc.).
4. **Retrieval**: When a user asks a question, embed the query and find the
   top-k most similar chunks.
5. **Augmentation**: Inject those chunks into the LLM prompt as context.
6. **Generation**: The LLM generates an answer grounded in the retrieved text.

## Why Chunking Matters

You can't just embed entire documents — they'd be too coarse. But tiny chunks
lose context. The sweet spot is usually 200-500 tokens per chunk, with some
overlap (50-100 tokens) to avoid cutting off ideas mid-sentence.

## Common Failure Modes

- **Poor retrieval**: The right chunk wasn't in top-k. Fix by improving
  chunking, re-ranking, or hybrid search (BM25 + vectors).
- **Context stuffing**: Too many irrelevant chunks confuse the LLM.
- **Outdated index**: Documents changed but the vector store didn't re-embed.

## When NOT to Use RAG

If the knowledge is stable and small, you might fine-tune the model instead.
If the task is pure reasoning (math, code), retrieval adds noise.
