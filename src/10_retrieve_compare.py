"""
Milestone 4 — Retrieval A/B test.

Queries the SAME 5 questions against two stores:
  v1: vector_store.json      (no prefix)
  v2: vector_store_v2.json   (search_document: prefix on chunks, search_query: on queries)

Side-by-side output lets you see whether prefixes fix the Python-venv failure.

Run: python src/10_retrieve_compare.py
(Requires: vector_store.json AND vector_store_v2.json)
"""

import json
from pathlib import Path
import ollama
import numpy as np


V1_PATH = Path(__file__).parent.parent / "vector_store.json"
V2_PATH = Path(__file__).parent.parent / "vector_store_v2.json"
QUERY_PREFIX = "search_query: "


def embed_raw(text: str) -> np.ndarray:
    r = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(r["embedding"])


def embed_query_prefixed(text: str) -> np.ndarray:
    r = ollama.embeddings(model="nomic-embed-text", prompt=QUERY_PREFIX + text)
    return np.array(r["embedding"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_store(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    for r in data:
        r["embedding"] = np.array(r["embedding"])
    return data


def retrieve(query_vec: np.ndarray, store: list[dict], top_k: int = 3) -> list[dict]:
    scored = []
    for r in store:
        score = cosine_similarity(query_vec, r["embedding"])
        scored.append({
            "score": score,
            "source": r["source"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
        })
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def print_hits(label: str, hits: list[dict], expected_source: str):
    print(f"  {label}")
    for rank, h in enumerate(hits, start=1):
        mark = "✅" if h["source"] == expected_source and rank == 1 else "  "
        src = f"{h['source']} (chunk {h['chunk_index']})"
        print(f"    {mark} #{rank}  {h['score']:.4f}  {src}")


def main():
    v1 = load_store(V1_PATH)
    v2 = load_store(V2_PATH)
    print(f"Loaded v1 ({len(v1)} chunks) and v2 ({len(v2)} chunks)\n")

    # (query, expected_source) pairs
    tests = [
        ("How does attention work in transformers?",       "transformer_architecture.md"),
        ("What is cosine similarity?",                      "embeddings_basics.md"),
        ("How do I create a Python virtual environment?",   "python_tips.md"),
        ("Why do we need chunking in RAG?",                 "rag_overview.md"),
        ("Which vector database should I use for production?", "vector_databases.md"),
    ]

    v1_correct = v2_correct = 0

    for query, expected in tests:
        print(f"🔍 Query: '{query}'")
        print(f"   Expected top-1 source: {expected}")

        # v1: no prefix anywhere
        v1_hits = retrieve(embed_raw(query), v1)
        print_hits("v1 (no prefix):", v1_hits, expected)
        if v1_hits[0]["source"] == expected:
            v1_correct += 1

        # v2: query uses search_query: prefix, chunks were embedded with search_document:
        v2_hits = retrieve(embed_query_prefixed(query), v2)
        print_hits("v2 (prefixed):", v2_hits, expected)
        if v2_hits[0]["source"] == expected:
            v2_correct += 1

        print()

    print("=" * 60)
    print(f"Top-1 accuracy: v1 = {v1_correct}/{len(tests)}   v2 = {v2_correct}/{len(tests)}")


if __name__ == "__main__":
    main()
