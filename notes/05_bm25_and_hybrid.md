# 05 — BM25 and Hybrid Search

**Date:** 2026-04-20
**Stage:** 2 (First LLM app — embeddings → RAG)
**Milestone:** Path A (Hybrid retrieval — BM25 + vectors)
**Scripts:** `src/12_bm25_from_scratch.py`, `src/13_hybrid_retrieve.py`
**Result:** Top-1 accuracy went from 4/5 (vectors alone) → 5/5 (hybrid)
**Builds on:** `notes/03_retrieval_and_prefixes.md`

---

## 1. What I Built / Learned

I implemented BM25 from scratch in ~30 lines of Python — the classic
keyword retrieval algorithm that's been the backbone of search engines for
20+ years — then validated my math against the `rank-bm25` library (ranks
matched perfectly, scores within expected variant-tolerance). I then built
hybrid retrieval that combines BM25 and cosine similarity scores with
min-max normalization and a blending weight α = 0.5. On our 5-query test
set, vectors alone scored 4/5, BM25 alone also scored 4/5 — but they
FAILED ON DIFFERENT QUERIES. Vectors missed "Python venv" (no semantic
cluster for keyword-y queries), BM25 missed "Chunking in RAG" (matched
common keywords in wrong chunks). Hybrid scored 5/5 because the two
methods cover each other's blind spots. This was my first visceral
experience of *why* modern retrieval systems are layered, not
monolithic.

---

## 2. The Child-Level Explanation

### 🧒 Child's first question
> "If embeddings already find meaning, why do I need keyword search too?
> Isn't keyword search the old, dumb way?"

**Answer:** Keyword search isn't obsolete — it's **different**. Embeddings
and keywords have **opposite failure modes**:

| | Embeddings | BM25 |
|---|---|---|
| Strength | Paraphrased/semantic queries ("how do I split text for search?") | Rare keywords ("venv", "HNSW", "FAISS") |
| Weakness | Specific rare terms the model never clusters well | Queries phrased without the exact document words |

When a user asks *"how do I create a Python virtual environment?"*,
they literally want the chunk that says `"python3 -m venv"`. Embeddings
generalize "virtual environment" into a vague "AI setup" region. BM25
sees *"venv"* appears once, in one chunk, in a 28-chunk corpus → high
IDF → massive score.

**The insight: hybrid = union of their strengths.**

---

## 3. BM25 — The Algorithm

### The formula (plain English)

For each query word, score = how often the word is in the doc × how rare
the word is across the corpus, then apply a length penalty.

### The actual formula per query term t, doc D:

```
score(t, D) = IDF(t) × (f(t, D) × (k1 + 1))
                       ────────────────────────────────────────────
                       f(t, D) + k1 × (1 - b + b × |D| / avg_doc_len)
```

Where:
- `f(t, D)` — how many times term t appears in doc D (term frequency)
- `IDF(t)` — rarity weight: `log((N - df + 0.5) / (df + 0.5) + 1)`
- `|D|` — number of tokens in doc D
- `avg_doc_len` — average doc length across corpus
- `k1 = 1.5` — **saturation**. After ~3 occurrences, more barely help.
  Prevents keyword stuffing.
- `b = 0.75` — **length normalization**. Discounts long docs so they don't
  cheat by containing every word by accident.

### The 4 forces inside the formula

1. **TF** (`f`) — more occurrences → higher score
2. **TF saturation** (`k1` in denominator) — diminishing returns on
   repetition
3. **Length penalty** (`b × |D| / avg_doc_len`) — long docs get discounted
4. **IDF** — rare terms contribute much more than common ones

### Why IDF is the magic

```python
self.idf = {
    term: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    for term, df in self.df.items()
}
```

- Common word like `"the"` appears in all 28 docs → IDF ≈ 0 → contributes
  nothing.
- Rare word like `"venv"` appears in 1 of 28 docs → IDF is HIGH → contributes
  a lot.

This is why BM25 latches onto specific keywords: rarity = signal.

---

## 4. The From-Scratch Implementation

The core class (simplified):

```python
class BM25FromScratch:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus_tokens = [tokenize(d) for d in corpus]
        self.doc_lengths = [len(t) for t in self.corpus_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths)
        self.N = len(corpus)

        # document frequency: how many docs contain each term
        df = Counter()
        for toks in self.corpus_tokens:
            for term in set(toks):           # set() ← count EACH doc once
                df[term] += 1

        # IDF — the rarity weight
        self.idf = {
            term: math.log((self.N - f + 0.5) / (f + 0.5) + 1)
            for term, f in df.items()
        }

    def score(self, query, doc_idx):
        query_terms = tokenize(query)
        doc_tokens = self.corpus_tokens[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        tf = Counter(doc_tokens)

        s = 0.0
        for term in query_terms:
            if term not in self.idf: continue
            f = tf.get(term, 0)
            if f == 0: continue
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            s += self.idf[term] * (num / den)
        return s
```

**Total: ~30 lines of real logic.** Everything the search engine giants
used for two decades, demystified.

### Validation against `rank-bm25`

On 3 test queries, my ranks matched the library's **perfectly**. Score
magnitudes differed slightly (I used IDF variant `log(1 + x)`, rank-bm25
uses `log(x)` for negatives — cosmetic differences). **Correctness
confirmed.**

---

## 5. The Hybrid Algorithm

```python
vec_norm = min_max(vector_scores)    # cosine → [0, 1]
bm_norm  = min_max(bm25_scores)      # BM25   → [0, 1]
hybrid   = ALPHA * vec_norm + (1 - ALPHA) * bm_norm
```

### Why normalization matters

Raw cosine scores range 0–1. Raw BM25 scores range 0–20+ (and are
unbounded). If we blended raw: `hybrid = 0.5 × 0.7 + 0.5 × 8.1 = 4.4` —
BM25 dominates purely because of scale, not quality.

**Min-max per query** puts both on `[0, 1]` so blending is fair:
```python
def min_max(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
```

### Why α = 0.5

Without a proper eval set yet, biasing either way is pre-judging. 0.5 is
the neutral starting point. In Path C (the 20-query eval) we'll tune α.

### Alternative: Reciprocal Rank Fusion (RRF)
A popular alternative that uses ranks instead of scores:
```
RRF(d) = Σ_methods  1 / (k + rank_method(d))    # k usually = 60
```
Avoids normalization entirely. Worth trying later.

---

## 6. What I Observed

### BM25 on the failing "Python venv" query
```
Rank  OURS score     source / preview
1     8.1362         python_tips.md (venv chunk)          ← correct at #1!
2     6.0062         embeddings_basics.md (king/queen)
3     5.3516         python_tips.md (code example)
```

**BM25 score 8.14 for the correct chunk.** Embeddings had buried this same
chunk at #3 with score 0.572. Completely different outcome from the same
data. Keyword search saw what semantic search missed.

### The final hybrid scoreboard
```
Query                          Vector  BM25   Hybrid
Attention                      ✅      ✅      ✅
Cosine similarity              ✅      ✅      ✅
Python venv                    ❌      ✅      ✅ 🚑 BM25 rescue
Chunking in RAG                ✅      ❌      ✅ 🚑 Vector rescue
Vector database                ✅      ✅      ✅

Top-1 accuracy:  Vector = 4/5   BM25 = 4/5   HYBRID = 5/5
```

### The two rescues in detail

**Rescue #1 — Python venv**
```
Vector:  ❌ rag_overview.md        (0.6160)
BM25:    ✅ python_tips.md (venv)  (8.1362)
Hybrid:  ✅ python_tips.md (venv)  (0.9995)
```
BM25's overwhelming confidence normalized to ~1.0, pulling it to top-1
in the blend.

**Rescue #2 — Chunking in RAG**
```
Vector:  ✅ rag_overview.md chunk 5      (0.6949)
BM25:    ❌ embeddings_basics.md chunk 1 (7.9469)   ← matched "chunks"/"embeddings" keywords wrongly
Hybrid:  ✅ rag_overview.md chunk 5      (0.8096)
```
BM25 got fooled by keyword co-occurrence, but the vector signal was
strong enough to hold the correct chunk at #1 after blending.

---

## 7. What Surprised Me

1. **BM25 and vectors disagreed on 2 out of 5 queries.** That's 40% —
   not rare. If retrieval methods agreed most of the time, hybrid would
   be pointless. The disagreement is precisely *where* hybrid earns its
   keep.

2. **BM25 is trivial — like 30 lines.** I expected a deep, scary
   algorithm given how foundational it is to all search. The whole thing
   is term frequency × inverse document frequency, with saturation and
   length penalties bolted on. **Classical ≠ complex.**

3. **Rank swap within correct territory is fine.** On the attention
   query, hybrid's #1 changed from "Self-Attention" (vec top-1) to
   "Multi-Head Attention" (hybrid top-1) — same file, different section.
   Both are valid top-1 answers. Worth noting but not a failure.

4. **Min-max normalization is almost criminally simple.** `(x - min) /
   (max - min)` — three tokens, saves the entire blending algorithm
   from the scale-mismatch catastrophe. The simplest good ideas always
   win.

---

## 8. What I Got Wrong at First

| Misconception | Correction |
|---|---|
| "BM25 is legacy — modern RAG uses only embeddings." | False. Every serious production RAG in 2026 uses hybrid. Embeddings + BM25 + reranker is the standard stack. |
| "Weighted sum of scores just works." | Not without normalization. BM25 0–20 + cosine 0–1 = BM25 dominates. Normalize or use RRF. |
| "If BM25 and vectors both get 4/5, hybrid should also get ~4/5." | Depends on *which* 4/5. If they fail on different queries, hybrid can hit 5/5. The fact that they failed on different queries is what makes hybrid work. |
| "The blending weight α is a minor knob." | Potentially critical. On keyword-heavy corpora, α → 0 (BM25 dominates). On paraphrase-heavy corpora, α → 1 (vectors dominate). Tune with evals. |

---

## 9. Follow-up Questions I Still Have

1. **Is there a principled way to pick α, or is it pure trial-and-error?**
2. **How does Reciprocal Rank Fusion compare to weighted sum in practice?**
   Is it strictly better?
3. **At what corpus size does BM25 start to lose to dense vectors?** I've
   heard dense search scales better — is that true?
4. **How do cross-encoders (rerankers) differ from BM25 for the "last
   mile" of retrieval quality?**
5. **Should I use different tokenization for BM25 vs embeddings?** I used
   simple `\w+` regex — production uses stemming, stop-word removal,
   etc. How much does that matter for small corpora?

---

## 10. Where This Fits in the Roadmap

- **Stage:** 2 (First LLM app — embeddings → RAG)
- **Completed:** Milestones 1–5 + Path A (hybrid retrieval — this note)
- **Up next:** Path C — the 20-query labeled eval set. Now that we have
  three retrieval modes (vector / BM25 / hybrid) plus end-to-end
  generation, we need rigorous measurement, not ad-hoc testing.

**Why this matters:** We've been testing with 5 hand-picked queries. Real
systems need diverse, labeled eval sets with categories (factoids,
paraphrases, rare keywords, synthesis, out-of-scope). Only then can we
answer questions like: "Is hybrid *really* better than vectors, or did we
cherry-pick 5 queries where it happens to win?" Evals turn guessing into
knowing.

---

## 11. Commands I Ran

```bash
pip install rank-bm25
python src/12_bm25_from_scratch.py    # validate our BM25 against library
python src/13_hybrid_retrieve.py      # A/B/C: vector vs BM25 vs hybrid
```
