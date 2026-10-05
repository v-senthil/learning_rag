"""
Milestone 4 — Path A: hybrid retrieval (BM25 + vectors).

For each query:
  1. Score every chunk with BM25           (keyword signal)
  2. Score every chunk with cosine sim     (semantic signal, prefixed)
  3. Min-max normalize both to [0, 1]
  4. Combined score = ALPHA * vector + (1 - ALPHA) * BM25
  5. Rank by combined score, compare to vector-only and BM25-only

Run: python src/13_hybrid_retrieve.py
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
import ollama
import numpy as np


STORE_PATH = Path(__file__).parent.parent / "vector_store_v2.json"
EMBED_MODEL = "nomic-embed-text"
QUERY_PREFIX = "search_query: "
TOP_K = 3
ALPHA = 0.5   # vector weight (1 - ALPHA = BM25 weight)


# ---------- tokenize (must match what we used during BM25 training) ----------
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ---------- BM25 (same class from Milestone 4, Path A) ----------
class BM25FromScratch:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus_tokens = [tokenize(d) for d in corpus]
        self.doc_lengths = [len(t) for t in self.corpus_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths)
        self.N = len(corpus)

        df = Counter()
        for toks in self.corpus_tokens:
            for term in set(toks):
                df[term] += 1
        self.idf = {
            t: math.log((self.N - f + 0.5) / (f + 0.5) + 1)
            for t, f in df.items()
        }

    def get_scores(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        scores = np.zeros(self.N)
        for i, (doc_toks, doc_len) in enumerate(zip(self.corpus_tokens, self.doc_lengths)):
            tf = Counter(doc_toks)
            s = 0.0
            for term in q_terms:
                if term not in self.idf:
                    continue
                f = tf.get(term, 0)
                if f == 0:
                    continue
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                s += self.idf[term] * (num / den)
            scores[i] = s
        return scores


# ---------- vector scoring ----------
def cosine_scores(query: str, store: list[dict]) -> np.ndarray:
    r = ollama.embeddings(model=EMBED_MODEL, prompt=QUERY_PREFIX + query)
    q = np.array(r["embedding"])
    q_norm = np.linalg.norm(q)
    out = np.zeros(len(store))
    for i, chunk in enumerate(store):
        v = chunk["embedding"]
        out[i] = float(np.dot(q, v) / (q_norm * np.linalg.norm(v)))
    return out


# ---------- normalization ----------
def min_max(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ---------- pretty printing ----------
def top_k_indices(scores: np.ndarray, k: int = TOP_K) -> list[int]:
    return list(np.argsort(-scores)[:k])


def print_column(label: str, scores: np.ndarray, store: list[dict], expected: str):
    idxs = top_k_indices(scores)
    print(f"  {label}")
    for rank, i in enumerate(idxs, 1):
        mark = "✅" if store[i]["source"] == expected and rank == 1 else "  "
        src = f"{store[i]['source']} (chunk {store[i]['chunk_index']})"
        print(f"    {mark} #{rank}  {scores[i]:.4f}  {src}")


def main():
    raw = json.loads(STORE_PATH.read_text())
    # convert embeddings to numpy up front (faster)
    for r in raw:
        r["embedding"] = np.array(r["embedding"])
    print(f"Loaded {len(raw)} chunks\n")

    # build BM25 once over the full corpus
    bm25 = BM25FromScratch([c["text"] for c in raw])

    tests = [
        ("How does attention work in transformers?",       "transformer_architecture.md"),
        ("What is cosine similarity?",                      "embeddings_basics.md"),
        ("How do I create a Python virtual environment?",   "python_tips.md"),
        ("Why do we need chunking in RAG?",                 "rag_overview.md"),
        ("Which vector database should I use for production?", "vector_databases.md"),
    ]

    wins = {"vector": 0, "bm25": 0, "hybrid": 0}

    for query, expected in tests:
        print(f"🔍 '{query}'")
        print(f"   Expected top-1: {expected}")

        vec = cosine_scores(query, raw)
        bm = bm25.get_scores(query)

        vec_norm = min_max(vec)
        bm_norm = min_max(bm)
        hybrid = ALPHA * vec_norm + (1 - ALPHA) * bm_norm

        print_column("Vector-only:", vec, raw, expected)
        print_column(f"BM25-only:  ", bm, raw, expected)
        print_column(f"HYBRID (α={ALPHA}):", hybrid, raw, expected)

        if raw[top_k_indices(vec)[0]]["source"] == expected:    wins["vector"] += 1
        if raw[top_k_indices(bm)[0]]["source"] == expected:     wins["bm25"] += 1
        if raw[top_k_indices(hybrid)[0]]["source"] == expected: wins["hybrid"] += 1
        print()

    print("=" * 60)
    print(f"Top-1 accuracy over {len(tests)} queries:")
    print(f"  Vector only:    {wins['vector']}/{len(tests)}")
    print(f"  BM25 only:      {wins['bm25']}/{len(tests)}")
    print(f"  HYBRID (α=0.5): {wins['hybrid']}/{len(tests)}")


if __name__ == "__main__":
    main()
