"""
Milestone 7 — Reusable Q&A tool.

Loads any store built by 17_ingest.py and runs hybrid retrieval +
grounded LLM generation. Two modes:

  1. Interactive REPL (default):
       python src/18_ask.py
       python src/18_ask.py --store research.json

  2. One-shot:
       python src/18_ask.py --query "What does the paper say about X?"
       python src/18_ask.py -s research.json -q "summarize section 3"
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
import ollama
import numpy as np


EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3-vl:235b-cloud"
QUERY_PREFIX = "search_query: "
TOP_K = 5
ALPHA = 0.5
DEFAULT_STORE = Path(__file__).parent.parent / "my_store.json"


# ---------- BM25 (from scratch, same as earlier milestones) ----------
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25FromScratch:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.corpus_tokens = [tokenize(d) for d in corpus]
        self.doc_lengths = [len(t) for t in self.corpus_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1
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
                if term not in self.idf or term not in tf:
                    continue
                f = tf[term]
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                s += self.idf[term] * (num / den)
            scores[i] = s
        return scores


# ---------- retrieval helpers ----------
def cosine_scores(q_vec: np.ndarray, store: list[dict]) -> np.ndarray:
    q_norm = np.linalg.norm(q_vec)
    out = np.zeros(len(store))
    for i, c in enumerate(store):
        v = c["embedding"]
        out[i] = float(np.dot(q_vec, v) / (q_norm * np.linalg.norm(v)))
    return out


def min_max(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)


def embed_query(text: str) -> np.ndarray:
    r = ollama.embeddings(model=EMBED_MODEL, prompt=QUERY_PREFIX + text)
    return np.array(r["embedding"])


# ---------- prompt ----------
PROMPT_TEMPLATE = """You are a helpful assistant answering questions using the context below.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question using the CONTEXT. You MAY synthesize, infer, and combine facts across chunks — the exact wording of the question does not need to appear in the context.
- Do NOT introduce facts that are not supported by the context.
- ADJACENT facts are NOT sufficient grounds to answer. If the context mentions related concepts but does not cover the specific task, entity, or concept asked about, REFUSE.
- If you refuse, respond EXACTLY with:
  "I don't have enough information to answer that based on the provided context."
- Cite sources in square brackets when you use a fact, e.g. [rag_overview.md].
- Be concise. 2-4 sentences.

REFUSAL EXAMPLE (for reference only — not the actual question):
  Question: "How do I train a language model on my laptop?"
  Hypothetical context: mentions GPUs, transformers, and parallelism — but not training.
  Correct answer: "I don't have enough information to answer that based on the provided context."
  (Reason: training is not covered; mentioning GPUs and transformers is adjacent, not sufficient.)

QUESTION: {query}

ANSWER:"""


def build_prompt(query: str, hits: list[dict]) -> str:
    ctx = "\n\n".join(
        f"[Source: {h['source']}]\n{h['text']}"
        for h in hits
    )
    return PROMPT_TEMPLATE.format(context=ctx, query=query)


def generate(prompt: str) -> str:
    r = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip()


# ---------- hybrid retrieval ----------
def retrieve(query: str, store: list[dict], bm25: BM25FromScratch,
             top_k: int = TOP_K, alpha: float = ALPHA) -> list[dict]:
    q_vec = embed_query(query)
    vec = cosine_scores(q_vec, store)
    bm = bm25.get_scores(query)
    hybrid = alpha * min_max(vec) + (1 - alpha) * min_max(bm)
    top_idx = np.argsort(-hybrid)[:top_k]

    hits = []
    for i in top_idx:
        hits.append({
            "score": float(hybrid[i]),
            "vec_score": float(vec[i]),
            "bm25_score": float(bm[i]),
            "source": store[i]["source"],
            "chunk_index": store[i]["chunk_index"],
            "text": store[i]["text"],
        })
    return hits


def ask(query: str, store: list[dict], bm25: BM25FromScratch) -> tuple[str, list[dict]]:
    hits = retrieve(query, store, bm25)
    prompt = build_prompt(query, hits)
    answer = generate(prompt)
    return answer, hits


# ---------- UX ----------
def print_result(query: str, answer: str, hits: list[dict]) -> None:
    print()
    print(f"❓ {query}")
    print()
    print(f"💡 {answer}")
    print()
    print("📄 Sources used:")
    for rank, h in enumerate(hits, 1):
        preview = h["text"][:90].replace("\n", " ")
        print(f"   {rank}. [hybrid {h['score']:.3f} | vec {h['vec_score']:.3f} | bm25 {h['bm25_score']:.2f}]")
        print(f"      {h['source']} #{h['chunk_index']} — {preview}...")
    print()


def interactive(store: list[dict], bm25: BM25FromScratch) -> None:
    print("💬 Interactive Q&A mode. Type your question, or 'quit' / Ctrl+C to exit.\n")
    try:
        while True:
            query = input("❓ ").strip()
            if not query:
                continue
            if query.lower() in {"quit", "exit", "q"}:
                break
            answer, hits = ask(query, store, bm25)
            print_result(query, answer, hits)
    except (KeyboardInterrupt, EOFError):
        pass
    print("\n👋 bye")


def main():
    p = argparse.ArgumentParser(description="Ask questions over a vector store built with 17_ingest.py")
    p.add_argument("--store", "-s", type=Path, default=DEFAULT_STORE,
                   help=f"store path (default: {DEFAULT_STORE.name})")
    p.add_argument("--query", "-q", type=str, default=None,
                   help="one-shot query (skip interactive mode)")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--alpha", type=float, default=ALPHA,
                   help="hybrid vector weight: 0=BM25 only, 1=vectors only (default 0.5)")
    args = p.parse_args()

    if not args.store.exists():
        print(f"❌ Store not found: {args.store}")
        print(f"   Build one first with:  python src/17_ingest.py <files>")
        return

    raw = json.loads(args.store.read_text())
    for r in raw:
        r["embedding"] = np.array(r["embedding"])
    print(f"Loaded {len(raw)} chunks from {args.store.name}")

    bm25 = BM25FromScratch([c["text"] for c in raw])

    if args.query:
        answer, hits = ask(args.query, raw, bm25)
        print_result(args.query, answer, hits)
    else:
        interactive(raw, bm25)


if __name__ == "__main__":
    main()
