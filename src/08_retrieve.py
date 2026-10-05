"""
Milestone 4 — Retrieval.

Load vector_store.json, embed a user query, score against every chunk,
return the top-k best matches. This is the "R" in RAG.

Run: python src/08_retrieve.py
"""

import json
from pathlib import Path
import ollama
import numpy as np


STORE_PATH = Path(__file__).parent.parent / "vector_store.json"


def embed(text: str) -> np.ndarray:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(response["embedding"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_store() -> list[dict]:
    """Load chunks + their embeddings from disk."""
    data = json.loads(STORE_PATH.read_text())
    # convert lists back to numpy for fast math
    for record in data:
        record["embedding"] = np.array(record["embedding"])
    return data


def retrieve(query: str, store: list[dict], top_k: int = 3) -> list[dict]:
    """Embed the query, score every chunk, return top-k sorted by similarity."""
    query_vec = embed(query)

    scored = []
    for record in store:
        score = cosine_similarity(query_vec, record["embedding"])
        scored.append({
            "score": score,
            "source": record["source"],
            "chunk_index": record["chunk_index"],
            "text": record["text"],
        })

    # sort descending by score, take top-k
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def pretty_print(query: str, results: list[dict]):
    print(f"\n🔍 Query: '{query}'\n")
    print("-" * 70)
    for rank, r in enumerate(results, start=1):
        preview = r["text"][:200].replace("\n", " ")
        print(f"#{rank}  score={r['score']:.4f}  source={r['source']} (chunk {r['chunk_index']})")
        print(f"     {preview}{'...' if len(r['text']) > 200 else ''}")
        print("-" * 70)


def main():
    store = load_store()
    print(f"Loaded {len(store)} chunks from {STORE_PATH.name}")

    # run several test queries to see retrieval quality
    queries = [
        "How does attention work in transformers?",
        "What is cosine similarity?",
        "How do I create a Python virtual environment?",
        "Why do we need chunking in RAG?",
        "Which vector database should I use for production?",
    ]

    for q in queries:
        results = retrieve(q, store, top_k=3)
        pretty_print(q, results)


if __name__ == "__main__":
    main()
