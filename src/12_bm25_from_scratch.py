"""
Milestone 4 — Path A: BM25 keyword search, from scratch.

Implements BM25 with no ML magic — just term frequency, inverse document
frequency, and length normalization. Then validates our math against the
rank-bm25 library on the same corpus.

Run: python src/12_bm25_from_scratch.py
"""

import json
import math
import re
from collections import Counter
from pathlib import Path


STORE_PATH = Path(__file__).parent.parent / "vector_store_v2.json"


# ---------- tokenizer (shared) ----------
def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric tokens. Same tokenizer for
    queries and docs — symmetry matters for BM25."""
    return re.findall(r"\w+", text.lower())


# ---------- BM25 from scratch ----------
class BM25FromScratch:
    """Minimal BM25 implementation.

    Score formula per query term:
        IDF(t) × f(t, D) × (k1 + 1) / (f(t, D) + k1 × (1 - b + b × |D|/avg_doc_len))

    - f(t, D): how many times term t appears in doc D  (term frequency)
    - IDF(t):  rarity weight; rare terms count more
    - |D|:     doc length; longer docs get penalized
    - k1, b:   tuning constants (standard defaults)
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = [tokenize(doc) for doc in corpus]
        self.doc_lengths = [len(toks) for toks in self.corpus_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths)
        self.N = len(corpus)

        # document frequency: how many docs contain each term at least once
        self.df = Counter()
        for tokens in self.corpus_tokens:
            for term in set(tokens):
                self.df[term] += 1

        # precompute IDF for every term — the "rarity weight"
        # formula: log((N - df + 0.5) / (df + 0.5) + 1)  — BM25's IDF variant
        self.idf = {
            term: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.df.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        query_terms = tokenize(query)
        doc_tokens = self.corpus_tokens[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        tf = Counter(doc_tokens)

        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue  # unseen term contributes nothing
            f = tf.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += self.idf[term] * (numerator / denominator)
        return score

    def get_scores(self, query: str) -> list[float]:
        return [self.score(query, i) for i in range(self.N)]


# ---------- compare our impl against rank-bm25 ----------
def compare(query: str, chunks: list[dict], top_k: int = 3):
    corpus_texts = [c["text"] for c in chunks]

    # Ours
    ours = BM25FromScratch(corpus_texts)
    our_scores = ours.get_scores(query)

    # Library (rank-bm25 wants pre-tokenized corpus)
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    lib = BM25Okapi(tokenized_corpus)
    lib_scores = lib.get_scores(tokenize(query))

    # rank top-k under each
    def top(scores):
        idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(i, scores[i]) for i in idxs]

    print(f"\n🔍 Query: '{query}'\n")
    print(f"{'Rank':<5} {'OURS score':<14} {'source / preview':<60}")
    for rank, (i, s) in enumerate(top(our_scores), 1):
        preview = chunks[i]["text"][:55].replace("\n", " ")
        print(f"{rank:<5} {s:<14.4f} {chunks[i]['source']:<30} {preview}...")

    print()
    print(f"{'Rank':<5} {'LIB  score':<14} {'source / preview':<60}")
    for rank, (i, s) in enumerate(top(lib_scores), 1):
        preview = chunks[i]["text"][:55].replace("\n", " ")
        print(f"{rank:<5} {s:<14.4f} {chunks[i]['source']:<30} {preview}...")


def main():
    chunks = json.loads(STORE_PATH.read_text())
    print(f"Loaded {len(chunks)} chunks\n")

    queries = [
        "How do I create a Python virtual environment?",   # the failure case!
        "What is cosine similarity?",
        "Which vector database should I use?",
    ]
    for q in queries:
        compare(q, chunks)
        print("=" * 70)


if __name__ == "__main__":
    main()
