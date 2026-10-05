# 03 — Retrieval and Query Prefixes

**Date:** 2026-04-20
**Stage:** 2 (First LLM app — embeddings → RAG)
**Milestone:** #4 of 7 (Retrieval / Vector Search)
**Scripts:** `src/08_retrieve.py`, `src/09_rebuild_with_prefixes.py`, `src/10_retrieve_compare.py`
**Outputs:** `vector_store.json` (v1, no prefix), `vector_store_v2.json` (v2, `search_document:` prefix)
**Builds on:** `notes/01_embeddings_and_similarity.md`, `notes/02_chunking.md`

---

## 1. What I Built / Learned

I implemented the "R" in RAG: a retrieval function that embeds a user query,
scores it against every chunk in the store using cosine similarity, and
returns the top-k best matches. I tested it with 5 hand-picked queries and
got 4/5 top-1 accuracy — one query (Python venv) failed because of a
question-vs-statement mismatch in embedding space. I then rebuilt the store
using nomic-embed-text's native `search_document:` prefix on chunks and
`search_query:` prefix on queries. Every top-1 score improved by ~0.02–0.03
(evidence the prefixes work), and the failed query climbed from rank #3 to
rank #2 — a partial rescue, not a full one. This taught me that fixes in
RAG compound: no single change gets you to 100%, and debugging requires
layered improvements.

---

## 2. The Child-Level Explanation

### 🧒 Child's first question
> "We have 28 chunks sitting in a JSON file. How does the computer know which
> one answers my question?"

**Answer:** It doesn't — until we compare. Retrieval works in 4 steps:

```
  user query
       ↓
  [embed the query]        ← same model as chunks, same 768-dim space
       ↓
  [cosine similarity]       ← score query vs every chunk
       ↓
  [sort descending]
       ↓
  [return top-k]            ← e.g., the top 3
```

That's it. **Retrieval is just "compute similarity, sort, slice."** No magic.

### 🔑 The most important rule — shared space
The query and the chunks MUST be embedded with the **same model**. Both live
in the same 768-dim meaning space. Mixing models → vectors point in
unrelated directions → cosine similarity returns garbage.

### 🧒 Why can't the query just be matched by keywords?
Because users ask in natural language. A query like *"how do I split docs
for search?"* might need the chunk titled *"Why Chunking Matters"* — no
shared keywords, but same topic. Embeddings bridge that gap. (Though as
we'll see later in `notes/05`, keywords DO matter for some queries —
hybrid search fixes that.)

---

## 3. The Question-vs-Statement Problem

When we tested 5 queries, one failed hard:

```
Query:  "How do I create a Python virtual environment?"
Best chunk: python_tips.md (contains "python3 -m venv")
What won:   embeddings_basics.md (generic AI intro) → score 0.5899
Correct chunk ranked #3 (score 0.5720)
```

Three failure reasons compounded:

1. **Bunched scores** (0.5899 → 0.5720 = noise) — weak retrieval signal.
2. **Training bias of nomic-embed-text** — trained on AI content, biased
   toward AI-flavored chunks for any technical query.
3. **Question-vs-statement asymmetry** — the query is *a question*, the
   chunk is *a statement*. Different "voice" → different regions of
   embedding space, even for the same topic.

### The fix — nomic-embed-text's native prefixes

The model was trained with asymmetric prefixes so queries and documents
map to paired regions:

| Prefix | Used for | Purpose |
|---|---|---|
| `search_document:` | Indexed chunks | "This is a passage of knowledge" |
| `search_query:` | User questions | "This is a question seeking an answer" |

During training, the model was taught that a `search_query:` vector should
point at the `search_document:` vector that answers it — even with
different phrasing. Without prefixes, the model has to guess what each text
is. With prefixes, we're **telling it**, which dramatically improves
question-to-passage alignment.

---

## 4. The Code

### `src/08_retrieve.py` — bare retrieval
```python
def retrieve(query, store, top_k=3):
    query_vec = embed(query)
    scored = []
    for record in store:
        score = cosine_similarity(query_vec, record["embedding"])
        scored.append({"score": score, **record})
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]
```
**4 steps. That's the whole retrieval algorithm.**

### `src/09_rebuild_with_prefixes.py` — re-embed with doc prefix
```python
DOC_PREFIX = "search_document: "

def embed_document(text):
    r = ollama.embeddings(
        model="nomic-embed-text",
        prompt=DOC_PREFIX + text,
    )
    return r["embedding"]
```
Only difference from before: we prepend the prefix string. The stored chunk
text is unchanged (so the LLM later sees clean text).

### `src/10_retrieve_compare.py` — A/B test
Queries both v1 and v2 stores with the same 5 questions, using raw text
for v1 and `search_query:` prefix for v2.

---

## 5. What I Observed

### Ad-hoc test (v1 — no prefix)
```
🔍 "How does attention work in transformers?"        → ✅ transformer_arch (0.6461)
🔍 "What is cosine similarity?"                       → ✅ embeddings_basics (0.7788)
🔍 "How do I create a Python virtual environment?"    → ❌ embeddings_basics (0.5899)  [correct at #3]
🔍 "Why do we need chunking in RAG?"                  → ✅ rag_overview (0.6746)
🔍 "Which vector database should I use for production?" → ✅ vector_databases (0.7185)

Top-1 accuracy: 4/5
```

### A/B comparison (v1 vs v2)
```
Query                          | v1 top-1  | v2 top-1 | Δ       | Status
Attention                      | 0.6461    | 0.6782   | +0.032  | ✅→✅
Cosine similarity              | 0.7788    | 0.7940   | +0.015  | ✅→✅
Python venv                    | 0.5899    | 0.6160   | +0.026  | ❌→❌ (but rank #3→#2)
Chunking in RAG                | 0.6746    | 0.6949   | +0.020  | ✅→✅
Vector database                | 0.7185    | 0.7466   | +0.028  | ✅→✅

Top-1 accuracy: v1 = 4/5, v2 = 4/5 (unchanged)
```

**My reaction:** "Top-1 didn't change? But EVERY score went up! And the
failing query moved from #3 to #2? Prefixes ARE doing something — just not
enough to close the gap for query 3."

---

## 6. What Surprised Me

1. **Prefixes lift ALL scores, not just failing ones.** Even the 4
   already-correct queries got +0.02 boosts. Evidence the model is more
   confident across the board when it knows query-vs-document intent.

2. **Top-1 accuracy can stay flat while rankings improve underneath.**
   Python venv's correct chunk climbed from #3 → #2 with a +0.044 score
   jump. Invisible in the top-1 metric but visible in rank. This taught
   me that top-1 is a *crude* measure — MRR (mean reciprocal rank) or
   recall@k reveal finer movements.

3. **nomic-embed-text is AI-biased.** Even with prefixes, a generic AI
   intro chunk beat a literal venv-command chunk for the Python query.
   The bias is baked into what the model considers "relevant." Prefixes
   can't undo training distribution.

4. **Retrieval debugging has a distinct flavor.** It's not "does my code
   work?" It's "does my model behave sensibly on this data?" The bug was
   never in the code — it was in the interaction between model, data, and
   query phrasing. This is the Debug Mindset from CLAUDE.md in action.

---

## 7. What I Got Wrong at First

| Misconception | Correction |
|---|---|
| "Top-1 accuracy is the main metric for retrieval." | It's *a* metric. Rank-sensitive ones (MRR, recall@k) reveal finer improvements a top-1 flip-count misses. |
| "A single fix (like prefixes) should close big retrieval gaps." | Fixes compound. Prefixes alone lifted scores but couldn't overcome the model's AI bias. Real RAG = layered techniques. |
| "The question and document should embed to the same vector if they mean the same thing." | Not for every model. Question-shaped vs passage-shaped text live in different regions unless the model was trained (or prompted) to align them. |
| "If I got 80% accuracy, I'm probably done." | The 20% is where the real work starts. Easy queries pass any retriever. The hard ones expose the real failure modes. |

---

## 8. Follow-up Questions I Still Have

1. **How do reranking models differ from embedding models?** I keep hearing
   "retrieve with embeddings, rerank with a cross-encoder." What does a
   cross-encoder actually look at that an embedding model doesn't?
2. **Are there embedding models less AI-biased?** Would a general-purpose
   model like `bge-large` or `e5` do better on the venv query?
3. **What's MRR exactly, and how would I compute it on my eval set?**
4. **How do production systems measure retrieval quality at scale?** Do
   they use human labels, LLM-as-judge, or something else?
5. **At what corpus size does JSON stop working and I need a real vector
   DB?** 100 chunks? 10k? 1M?

---

## 9. Where This Fits in the Roadmap

- **Stage:** 2 (First LLM app — embeddings → RAG)
- **Completed so far:**
  - ✅ Milestone 1: setup + sample data
  - ✅ Milestone 2: embeddings + similarity (`notes/01`)
  - ✅ Milestone 3: chunking + mini vector store (`notes/02`)
  - ✅ Milestone 4: retrieval + query prefixes (this note)
- **Up next:** Milestone 5 — end-to-end RAG. We'll wire this retrieval into
  an LLM so it produces grounded answers with citations.

**Why this matters:** Retrieval is the gate that decides what knowledge the
LLM sees. If retrieval is wrong, generation is wrong — no matter how smart
the LLM is. **Garbage in, garbage out.** Every hour invested in retrieval
quality pays back 10x in answer quality.

---

## 10. Commands I Ran

```bash
python src/08_retrieve.py                # bare retrieval (v1, no prefix)
python src/09_rebuild_with_prefixes.py   # rebuild store with search_document:
python src/10_retrieve_compare.py        # v1 vs v2 A/B test
```
