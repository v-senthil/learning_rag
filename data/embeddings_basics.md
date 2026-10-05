# Embeddings: The Foundation of Modern AI

## What Are Embeddings?

An embedding is a numerical representation of data (text, images, audio) as a
vector of floating-point numbers. The key idea is that **similar things end up
close together in this vector space**.

For example, the words "king" and "queen" will have embeddings that are
mathematically close, while "king" and "banana" will be far apart.

## Why Do We Need Them?

Computers don't understand words — they understand numbers. Embeddings are how
we translate human language into a form a machine can reason about.

Before embeddings, we used techniques like one-hot encoding, which wasted space
and captured no meaning. Embeddings are dense, compact, and semantic.

## Typical Dimensions

- Small models: 384 dimensions (e.g., all-MiniLM-L6-v2)
- Medium models: 768 dimensions (e.g., nomic-embed-text)
- Large models: 1536+ dimensions (e.g., OpenAI text-embedding-3-large)

More dimensions = more nuance, but also more storage and compute cost.

## How Similarity Is Measured

The most common metric is **cosine similarity**, which measures the angle
between two vectors. A score of 1.0 means identical direction (very similar),
0.0 means orthogonal (unrelated), and -1.0 means opposite.

## Common Use Cases

- Semantic search
- Recommendation systems
- Clustering documents
- Retrieval-Augmented Generation (RAG)
- Anomaly detection
