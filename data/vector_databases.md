# Vector Databases

## What They Do

A vector database stores high-dimensional vectors (embeddings) and lets you
perform **approximate nearest neighbor (ANN)** search blazingly fast. Instead
of comparing a query to every stored vector (O(n)), they use clever indexing
to make search sub-linear.

## Popular Options

### FAISS (Facebook AI Similarity Search)
- In-memory, extremely fast, C++ under the hood.
- Great for prototyping and research.
- No built-in persistence or metadata filtering.

### Chroma
- Open-source, Python-first, easy to use.
- Built-in persistence and metadata support.
- Great for getting started quickly.

### Pinecone
- Fully managed cloud service.
- Handles scaling and reliability for you.
- Paid, but simple API.

### Weaviate
- Open-source with hybrid search (vector + keyword).
- GraphQL API, strong schema support.

### Qdrant
- Rust-based, fast, Docker-friendly.
- Excellent filtering and payload support.

## Key Concepts

- **Index type**: Flat (exact), HNSW (graph-based, very popular), IVF (inverted
  file, good for huge datasets).
- **Metric**: Cosine, dot product, or Euclidean distance.
- **Recall vs Speed**: ANN is approximate — you trade some accuracy for speed.
  Tune parameters based on your tolerance.

## Choosing One

- Learning / prototyping: Chroma or FAISS
- Production, managed: Pinecone
- Self-hosted production: Qdrant or Weaviate
- Huge scale (billions of vectors): FAISS with IVF+PQ indexes
