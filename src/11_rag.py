"""
Milestone 5 — End-to-end RAG.

Flow:
  user query → embed (with search_query: prefix) → retrieve top-3 chunks
             → build prompt → Llama 3.2 answers with citations.

Run: python src/11_rag.py
"""

import json
from pathlib import Path
import ollama
import numpy as np


STORE_PATH = Path(__file__).parent.parent / "vector_store_v2.json"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3-vl:235b-cloud"
QUERY_PREFIX = "search_query: "
TOP_K = 5


# ---------- retrieval (from Milestone 4) ----------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_store() -> list[dict]:
    data = json.loads(STORE_PATH.read_text())
    for r in data:
        r["embedding"] = np.array(r["embedding"])
    return data


def retrieve(query: str, store: list[dict], top_k: int = TOP_K) -> list[dict]:
    r = ollama.embeddings(model=EMBED_MODEL, prompt=QUERY_PREFIX + query)
    query_vec = np.array(r["embedding"])

    scored = []
    for chunk in store:
        score = cosine_similarity(query_vec, chunk["embedding"])
        scored.append({
            "score": score,
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
        })
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


# ---------- prompt construction ----------
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
    context_block = "\n\n".join(
        f"[Source: {h['source']}]\n{h['text']}"
        for h in hits
    )
    return PROMPT_TEMPLATE.format(context=context_block, query=query)


# ---------- LLM generation ----------
def generate(prompt: str) -> str:
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


# ---------- the full RAG loop ----------
def ask(query: str, store: list[dict]) -> None:
    print(f"\n🔍 Query: {query}")
    print("-" * 70)

    hits = retrieve(query, store)

    print("📄 Retrieved chunks:")
    for i, h in enumerate(hits, start=1):
        preview = h["text"][:80].replace("\n", " ")
        print(f"   {i}. [{h['score']:.3f}] {h['source']} → {preview}...")

    prompt = build_prompt(query, hits)
    answer = generate(prompt)

    print(f"\n💬 Answer:\n{answer}")
    print("=" * 70)


def main():
    store = load_store()
    print(f"Loaded {len(store)} chunks from {STORE_PATH.name}")

    queries = [
        "How does attention work in transformers?",
        "What is cosine similarity?",
        "Why do we need chunking in RAG?",
        # 👇 outside the knowledge base — tests whether the LLM refuses to guess
        "What is the capital of France?",
    ]
    for q in queries:
        ask(q, store)


if __name__ == "__main__":
    main()
