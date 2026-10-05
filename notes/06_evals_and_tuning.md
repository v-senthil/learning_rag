# 06 — Evals and Tuning (the real engineering starts here)

**Date:** 2026-04-20
**Stage:** 2 (First LLM app — embeddings → RAG)
**Milestone:** Path C (20-query eval set + iterative tuning)
**Files:** `eval_set.json` (23 queries, 5 categories), `src/14_eval_runner.py`
**Configurations tested:** 4 (top_k ∈ {3, 5, 4} × prompt ∈ {flexible, sharpened})
**Builds on:** `notes/05_bm25_and_hybrid.md`

---

## 1. What I Built / Learned

I built a 23-query labeled eval set spanning 5 categories (direct factoids,
paraphrases, rare keywords, cross-file synthesis, out-of-scope refusal traps)
and an eval runner that scores every retrieval mode (vector, BM25, hybrid)
plus end-to-end generation with the LLM. Then I used the eval set to tune
the system across 4 configurations, watching the numbers move. The biggest
lesson: **every tuning knob has two-sided pain.** Looser prompt = more
in-scope correct BUT more OOS leaks. More top-k = better recall BUT more
adjacent-context temptation for the LLM to synthesize OOS answers. Evals
made the invisible tradeoffs visible, and turned guessing into measured
decision-making. Without evals, I'd have declared "victory" at 15/19
(79%) — the eval set showed there was room to 18/19 (95%), and also showed
the cost of getting there (1 OOS leak). Now I can make the tradeoff
consciously instead of pretending it doesn't exist.

---

## 2. The Child-Level Explanation

### 🧒 Child's first question
> "Why do I need an eval set? I already ran 5 test queries — can't I just
> look at the answers and decide if they're good?"

**Answer:** Because 5 test queries is *sampling*, not measuring. A couple
problems with ad-hoc testing:

1. **Confirmation bias** — you write the test queries thinking of answers
   you know are in the docs. You're testing what you know, not what users
   will actually ask.
2. **No categories** — a passing rate of 4/5 could be "4/5 factoids" (easy)
   or "4/5 paraphrases" (hard). Without categorization, you don't know
   which kinds of queries your system handles well.
3. **No refusal testing** — ad-hoc queries are almost all answerable. The
   hardest RAG behavior (refusing out-of-scope) goes untested.
4. **No regression detection** — if you tweak something and re-run 5
   queries, you only see the headline. You miss subtle shifts.

### What a real eval set does
- Labels **many queries** (20+) covering diverse query shapes
- Assigns each a **category** (so you see where you're weak)
- Includes **traps** (OOS queries that should be refused)
- Produces a **deterministic scorecard** so you can compare across runs

### The loop evals unlock
```
 Build feature  →  Run eval  →  See the score  →  Decide next move  →  repeat
```
Without the "see the score" step, you're just hoping.

---

## 3. The Eval Set Design

**23 queries, 5 categories:**

| Category | Count | Purpose |
|---|---|---|
| `direct_factoid` | 5 | Literal lookups — every retriever should nail these |
| `paraphrase` | 5 | Semantic queries with NO shared keywords — vectors excel |
| `rare_keyword` | 5 (incl. 1 trap) | Specific terms like HNSW, FAISS, venv — BM25 excels |
| `cross_file_synthesis` | 5 | Requires combining info across chunks/files — good hybrid test |
| `out_of_scope` | 3 | Must be refused — tests grounding discipline |

### Each query has this schema:
```json
{
  "id": "KEY-03",
  "query": "What is BPE?",
  "expected_sources": ["embeddings_basics.md"],
  "expected_refusal": true,
  "category": "rare_keyword",
  "notes": "BPE is in my notes but NOT in data/. Must refuse."
}
```

- `expected_sources` — filenames where the answer should come from (empty for OOS)
- `expected_refusal` — `true` if the system should say "I don't know"
- `category` — for per-category breakdown metrics
- `notes` — why this query is interesting (for future self)

### The sneaky trap — KEY-03
Asks *"What is BPE?"* Mentioned in my learning notes, but NOT in
`data/*.md`. The system should refuse even though "BPE" sounds like it
belongs. This tests whether we can distinguish our knowledge base from
adjacent concepts we happen to know about.

---

## 4. Eval Runner Design

**Metrics computed per run:**

| Metric | What it measures |
|---|---|
| `top_1_source_accuracy` | Did the top-1 chunk come from an expected source? |
| `top_3_source_recall` | Did ANY of the top-k chunks come from expected sources? |
| `refusal_correct` | For OOS queries: did the LLM produce the exact refusal phrase? |
| `answered_correct` | For in-scope queries: did the LLM answer (not refuse)? |
| `per_category_breakdown` | Above metrics, sliced by category |

**CLI options:**
```bash
python src/14_eval_runner.py               # retrieval only (fast)
python src/14_eval_runner.py --with-gen    # adds LLM generation
python src/14_eval_runner.py --alpha 0.7   # tune hybrid blending
```

### Why retrieval-only is the default
Because generation is slow (seconds per query × 23 queries). During
iteration, you run retrieval-only to see scorecard movement fast. Flip
on `--with-gen` only for the "is this actually a good change?" moment.

---

## 5. The Four Configurations Tested

We iterated through 4 settings, each measured on the same 23-query set:

| # | Config | Key change |
|---|---|---|
| 1 | `k=3`, strict prompt | Starting baseline |
| 2 | `k=3`, flexible prompt | Softened "ONLY use context" |
| 3 | `k=5`, flexible prompt | More context per query |
| 4 | `k=5`, sharpened prompt | Added few-shot refusal example + "ADJACENT not sufficient" rule |

### Retrieval scorecard (identical across runs — retrieval is deterministic)
```
             Top-1           Top-3 recall (k=5)
vector       15/19 (79%)     19/19 (100%)
bm25         13/19 (68%)     19/19 (100%)
hybrid       17/19 (89%)     19/19 (100%)   ← best
```

### Per-category hybrid top-1
```
direct_factoid        5/5   ← perfect
paraphrase            3/5   ← weakest category (PARA-01, PARA-05 miss)
rare_keyword          4/4   ← perfect (excluding OOS-flagged KEY-03)
cross_file_synthesis  5/5   ← perfect
```

### Generation scorecard across the four runs
```
Config                  In-scope correct  OOS refusal  Overall
#1 k=3, strict          13/19 (68%)       4/4 (100%)   17/23 (74%)
#2 k=3, flexible        15/19 (79%)       4/4 (100%)   19/23 (83%)
#3 k=5, flexible        18/19 (95%)       3/4 (75%)    21/23 (91%)
#4 k=5, sharpened       16/19 (84%)       4/4 (100%)   20/23 (87%)
```

---

## 6. The Trade-Off We Discovered

The runs are not strictly ordered best→worst. Each configuration optimizes
a different thing. Side-by-side:

| Run | Strengths | Weaknesses |
|---|---|---|
| #2 (k=3 flexible) | Solid refusal discipline | Under-answers (79% in-scope) |
| #3 (k=5 flexible) | Highest in-scope (95%) | Leaks an OOS answer |
| #4 (k=5 sharpened) | Solid refusal AND higher in-scope | Over-refuses on a couple synthesis queries |

**The fundamental trade-off:**
```
           Looser prompt / more top-k
                       │
           Higher in-scope recall
                       │
                       ▼
               MORE OOS LEAKS
                       ▲
                       │
           Lower in-scope recall
                       │
           Stricter prompt / less top-k
```

### What this means in practice
- **User-facing chatbots:** lean strict. An OOS leak breaks trust — the
  user can't tell grounded answers from hallucinated ones. Take the hit
  on over-refusal.
- **Internal research tools:** lean flexible. Users know the system's
  limits and can re-ask. An over-refusal is just an annoyance.
- **No universal "best":** the right config depends on the use case.

---

## 7. The Most Important Finding

**Top-3 recall is 100% across all retrievers at k=5.** In plain English:
**for every in-scope query, the right chunk is ALWAYS in the prompt.**

That means **retrieval is no longer the bottleneck**. Every remaining
failure is a generation issue — either the LLM refuses when it
shouldn't, or answers when it shouldn't. Knowing this reframes the
debugging:
- Before: "Is my chunking right? My embeddings? My BM25 weights?"
- After: "The right chunks are arriving. The problem is how the LLM
  interprets them."

Your CLAUDE.md debug mindset says:
> *"Retrieval failure vs generation failure?"*

Evals are how you tell the difference. Before this, we were guessing.

---

## 8. The Prompt Tuning Story (in detail)

### Prompt v1 — strict (the initial version)
```
- Answer ONLY using information from the CONTEXT above.
- If the context does not contain the answer, respond EXACTLY with: "I don't
  have enough information to answer that based on the provided context."
```
Result: 13/19 in-scope (68%). LLM was over-refusing queries whose answers
were present but needed light synthesis.

### Prompt v2 — flexible
```
- Answer the question using the CONTEXT. You MAY synthesize, infer, and
  combine facts across chunks — the exact wording of the question does not
  need to appear in the context.
- Do NOT introduce facts not supported by the context.
- ONLY refuse if the context is GENUINELY UNRELATED to the question...
```
Result at k=5: 18/19 in-scope (95%) BUT 3/4 OOS (leaked OOS-03 —
synthesized an answer about training LLMs from adjacent GPU/transformer
chunks).

### Prompt v3 — sharpened with few-shot
Added rule:
```
- ADJACENT facts are NOT sufficient grounds to answer. If the context
  mentions related concepts but does not cover the specific task, entity,
  or concept asked about, REFUSE.
```
Plus a refusal example:
```
REFUSAL EXAMPLE (for reference only — not the actual question):
  Question: "How do I train a language model on my laptop?"
  Hypothetical context: mentions GPUs, transformers, and parallelism — but not training.
  Correct answer: "I don't have enough information to answer that based on the provided context."
```
Result: OOS-03 now refuses ✅, but SYN-04 and SYN-05 broke — the new
"ADJACENT" rule is bleeding into real synthesis cases and causing
over-refusal.

### The lesson
Each prompt edit fixed the thing it targeted AND introduced new
failures elsewhere. No prompt is universally better. You have to pick
which failure mode you prefer to carry.

---

## 9. What Surprised Me

1. **Evals reveal things you can't see by reading answers.** Answer
   previews look fine individually. Only the aggregate scorecard exposes
   "this change fixed 2 queries but broke 2 others." A/B discipline
   over reading.

2. **Top-3 recall = 100% was an unexpected win.** I thought retrieval
   would still be our bottleneck. The fact that retrieval is "solved"
   (for this corpus) reroutes all future debugging effort to generation.

3. **Prompt-tuning is lossy.** I thought better prompts = strictly
   better results. Reality: better prompts = different results. Every
   improvement shifts the failure set, rarely shrinks it cleanly.

4. **The OOS refusal is a stronger teaching signal than in-scope
   accuracy.** Losing 1 OOS (4→3) felt worse than gaining 3 in-scope
   (15→18), because trust breaks asymmetrically. You can recover from
   being unhelpful; you can't recover from being confidently wrong.

5. **The eval set is reusable forever.** Every future change — new
   model, new chunking strategy, reranker, etc. — gets scored against
   the same 23 queries. I can now answer "is X change better?" with
   numbers instead of vibes.

---

## 10. What I Got Wrong at First

| Misconception | Correction |
|---|---|
| "If 3 test queries pass, my system works." | 3 queries is noise. You need 20+ spanning failure-prone categories to measure anything useful. |
| "Better prompt = better results, period." | Better prompt = different results. It shifts what breaks. Measure both axes (in-scope AND OOS) every time. |
| "The right top_k is a universal constant." | It depends on chunk size, corpus size, and how much the LLM can handle. 3-5 is typical but needs per-domain tuning. |
| "OOS queries are easy — the LLM just looks for the answer and can't find it." | The hardest queries. Adjacent context tempts synthesis. A well-prompted strong LLM is the ONLY reliable refusal mechanism. |
| "Overall accuracy is the single right metric." | Different failures hurt differently. An OOS leak is worse than an over-refusal for user-facing systems. Decompose the metric. |

---

## 11. Follow-up Questions I Still Have

1. **How big should an eval set be?** 23 feels small. Is there a rule of
   thumb for minimum eval size relative to corpus size?
2. **How do I decide whether to add a new query to the eval vs. fix code
   first?** Sometimes an unexpected failure could be symptom of a
   whole class of failures I'm not testing.
3. **Can I use an LLM to grade answers instead of just checking refusals?**
   LLM-as-judge would let me evaluate answer *correctness*, not just
   "did it answer?"
4. **When does it make sense to test multiple prompt versions as an
   "ensemble"?** Like run 3 prompts, pick the answer with most
   agreement.
5. **How much of this scorecard is really LLM-dependent?** If I swap
   qwen3-vl for Claude or GPT-4, do the trade-offs look different? (My
   guess: yes, different LLMs have different refusal thresholds.)

---

## 12. Where This Fits in the Roadmap

- **Stage:** 2 (First LLM app — embeddings → RAG)
- **Completed:**
  - ✅ Milestones 1–5: setup → retrieval → end-to-end RAG
  - ✅ Path A: hybrid retrieval
  - ✅ Path C: eval set + iterative tuning (this note)
- **Up next (Milestone 6):** Visualization — UMAP projection of the
  embedding space + Streamlit retrieval debugger UI. Now that we
  *understand* what the system does (via evals), time to *see* what
  the embedding space looks like.

**Why this matters:** Everything up to and including this note is the
"engineering" half of building RAG — measuring, tuning, debugging.
Visualization is the "understanding" half — seeing WHY clusters form,
watching queries navigate the embedding space, spotting outlier
chunks visually. Both halves compound.

---

## 13. The Data Snapshot (for future reference)

### Final retrieval hybrid (α=0.5, 23 queries)
```
direct_factoid         5/5   (100%)
paraphrase             3/5   (60%)   ← weakest
rare_keyword           4/4   (100%)  (KEY-03 excluded as OOS trap)
cross_file_synthesis   5/5   (100%)
overall in-scope       17/19 (89%)
top-3 recall           19/19 (100%)
```

### Final generation (k=5, sharpened prompt)
```
In-scope correct       16/19 (84%)
OOS refusal correct    4/4   (100%)
Overall                20/23 (87%)
```

### Queries still failing generation
| ID | Category | Why |
|---|---|---|
| FACT-04 | direct_factoid | LLM sees stages but won't commit to "six" — chunk boundaries likely split the list |
| SYN-04 | cross_file_synthesis | Over-triggered on "ADJACENT" rule |
| SYN-05 | cross_file_synthesis | Same as SYN-04 |

These are the next targets (reranker, better chunking at list
boundaries, or prompt rebalancing).

---

## 14. Commands I Ran

```bash
# main eval runs (one per configuration)
python src/14_eval_runner.py                   # retrieval only, fast
python src/14_eval_runner.py --with-gen        # full eval
python src/14_eval_runner.py --alpha 0.3       # more BM25 weight
python src/14_eval_runner.py --alpha 0.7       # more vector weight

# TOP_K was edited in the script: 3 → 5 → 4
# PROMPT_TEMPLATE was edited in the script across 3 versions
```
