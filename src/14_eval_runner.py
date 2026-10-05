"""
Milestone — Path C: eval runner.

Runs every query from eval_set.json through 3 retrieval modes
(vector-only, BM25-only, hybrid). Optionally runs end-to-end generation
to check refusal behavior on out-of-scope queries.

Metrics computed (per mode):
  - top_1_source_accuracy: did the top-1 chunk come from an expected source?
  - top_3_source_recall:   did ANY of the top-3 chunks come from an expected source?
  - refusal_accuracy (generation only): did the LLM correctly refuse on
    out-of-scope queries?
  - per-category breakdown

Run (retrieval only, fast):
  python src/14_eval_runner.py

Run (retrieval + generation):
  python src/14_eval_runner.py --with-gen

Run with custom alpha:
  python src/14_eval_runner.py --alpha 0.7
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import ollama
import numpy as np


STORE_PATH = Path(__file__).parent.parent / "vector_store_v2.json"
EVAL_PATH = Path(__file__).parent.parent / "eval_set.json"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3-vl:235b-cloud"
QUERY_PREFIX = "search_query: "
TOP_K = 5
REFUSAL_PHRASE = "I don't have enough information to answer that"


# ---------- tokenize ----------
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ---------- BM25 ----------
class BM25FromScratch:
    def __init__(self, corpus, k1=1.5, b=0.75):
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
                if term not in self.idf or term not in tf:
                    continue
                f = tf[term]
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


def min_max(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)


# ---------- metrics ----------
def top_1_correct(top_indices: list[int], store: list[dict], expected: list[str]) -> bool:
    if not expected:
        return False  # out-of-scope — retrieval "correctness" doesn't apply
    return store[top_indices[0]]["source"] in expected


def top_3_recall(top_indices: list[int], store: list[dict], expected: list[str]) -> bool:
    if not expected:
        return False
    return any(store[i]["source"] in expected for i in top_indices)


# ---------- generation ----------
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


def generate(query: str, top_indices: list[int], store: list[dict]) -> str:
    ctx = "\n\n".join(
        f"[Source: {store[i]['source']}]\n{store[i]['text']}"
        for i in top_indices
    )
    prompt = PROMPT_TEMPLATE.format(context=ctx, query=query)
    r = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip()


def is_refusal(answer: str) -> bool:
    return REFUSAL_PHRASE.lower() in answer.lower()


# ---------- main eval loop ----------
def run_eval(alpha: float, with_gen: bool):
    store = json.loads(STORE_PATH.read_text())
    for r in store:
        r["embedding"] = np.array(r["embedding"])

    eval_data = json.loads(EVAL_PATH.read_text())
    queries = eval_data["queries"]
    print(f"Loaded {len(store)} chunks, {len(queries)} eval queries (α={alpha}, with_gen={with_gen})\n")

    bm25 = BM25FromScratch([c["text"] for c in store])

    modes = ["vector", "bm25", "hybrid"]
    top1 = {m: 0 for m in modes}
    top3 = {m: 0 for m in modes}
    per_cat = defaultdict(lambda: {m: {"top1": 0, "top3": 0, "n": 0} for m in modes})
    gen_results = []

    in_scope_count = 0

    for q in queries:
        query, expected = q["query"], q["expected_sources"]
        cat = q["category"]
        is_oos = q["expected_refusal"]

        vec = cosine_scores(query, store)
        bm = bm25.get_scores(query)
        hyb = alpha * min_max(vec) + (1 - alpha) * min_max(bm)

        ranked = {
            "vector": list(np.argsort(-vec)[:TOP_K]),
            "bm25":   list(np.argsort(-bm)[:TOP_K]),
            "hybrid": list(np.argsort(-hyb)[:TOP_K]),
        }

        if not is_oos:
            in_scope_count += 1
            per_cat[cat]["_n"] = per_cat[cat].get("_n", 0) + 1
            for m in modes:
                if top_1_correct(ranked[m], store, expected):
                    top1[m] += 1
                    per_cat[cat][m]["top1"] += 1
                if top_3_recall(ranked[m], store, expected):
                    top3[m] += 1
                    per_cat[cat][m]["top3"] += 1
                per_cat[cat][m]["n"] += 1

        if with_gen:
            # generate using HYBRID retrieval (best of 3)
            answer = generate(query, ranked["hybrid"], store)
            refused = is_refusal(answer)
            if is_oos:
                correct = refused
            else:
                correct = not refused
            gen_results.append({
                "id": q["id"], "category": cat, "is_oos": is_oos,
                "refused": refused, "generation_correct": correct,
                "answer_preview": answer[:120].replace("\n", " "),
            })

        status = []
        for m in modes:
            ok1 = "✅" if top_1_correct(ranked[m], store, expected) else "❌"
            status.append(f"{m[0]}{ok1}")
        print(f"  [{q['id']}] {cat:<22} {' '.join(status)}  {query[:60]}")

    print("\n" + "=" * 70)
    print(f"RETRIEVAL SCORECARD ({in_scope_count} in-scope queries)")
    print(f"{'Mode':<10} {'Top-1':<12} {'Top-3 recall':<16}")
    for m in modes:
        p1 = top1[m] / in_scope_count
        p3 = top3[m] / in_scope_count
        print(f"{m:<10} {top1[m]}/{in_scope_count} ({p1:.0%})     {top3[m]}/{in_scope_count} ({p3:.0%})")

    print("\nPER-CATEGORY TOP-1 BREAKDOWN")
    for cat, vals in per_cat.items():
        n = vals["vector"]["n"]
        print(f"  {cat:<25} (n={n})  vec={vals['vector']['top1']}/{n}  bm25={vals['bm25']['top1']}/{n}  hyb={vals['hybrid']['top1']}/{n}")

    if with_gen:
        print("\nGENERATION SCORECARD")
        total = len(gen_results)
        oos = [g for g in gen_results if g["is_oos"]]
        in_s = [g for g in gen_results if not g["is_oos"]]
        print(f"  Total: {total}")
        print(f"  Refusal correct (out-of-scope): {sum(1 for g in oos if g['generation_correct'])}/{len(oos)}")
        print(f"  Answered correctly (in-scope):  {sum(1 for g in in_s if g['generation_correct'])}/{len(in_s)}")
        print("\nGeneration details:")
        for g in gen_results:
            tag = "OOS" if g["is_oos"] else "IN "
            mark = "✅" if g["generation_correct"] else "❌"
            refused_tag = "[REFUSED]" if g["refused"] else "[ANSWERED]"
            print(f"  {mark} [{g['id']}] {tag} {refused_tag} {g['answer_preview']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.5, help="hybrid vector weight (0=BM25 only, 1=vector only)")
    p.add_argument("--with-gen", action="store_true", help="also run LLM generation (slower)")
    args = p.parse_args()
    run_eval(args.alpha, args.with_gen)


if __name__ == "__main__":
    main()
