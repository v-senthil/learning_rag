# 04 — End-to-End RAG (Retrieval + Generation + Refusal)

**Date:** 2026-04-20
**Stage:** 2 (First LLM app — embeddings → RAG)
**Milestone:** #5 of 7 (RAG Pipeline)
**Script:** `src/11_rag.py`
**LLM used:** `qwen3-vl:235b-cloud` (via Ollama)
**Builds on:** `notes/03_retrieval_and_prefixes.md`

---

## 1. What I Built / Learned

I wired retrieval into an LLM and got a working end-to-end RAG system. For
the first time, user questions produced grounded answers with inline source
citations like `[rag_overview.md]`. More importantly, I tested the system
with an out-of-knowledge question ("What is the capital of France?") and it
correctly refused to answer — proving the LLM is following my prompt rules
rather than drawing on its training memory. That single refusal is what
separates a *toy* RAG (cosmetic retrieval bolted to a chatty model) from
a *real* RAG (grounded, auditable, safe). I also learned that prompt
design IS the system: every rule, every format instruction, every
refusal phrase was a deliberate choice that shaped behavior.

---

## 2. The Child-Level Explanation

### 🧒 Child's first question
> "If the LLM already knows things, why do we even need retrieval?"

**Two reasons:**

1. **The LLM's knowledge is frozen at training time.** It can't know about
   documents that didn't exist when it was trained — your company's wiki,
   yesterday's meeting notes, a new research paper.
2. **LLMs hallucinate.** When they don't know something, they often make
   up plausible-sounding answers rather than admit ignorance. RAG fixes
   both: we inject trusted, up-to-date context, and we *order* the model
   to stay inside that context.

### The full pipeline
```
  user query
       ↓
  [embed with search_query: prefix]   ← Milestone 4
       ↓
  [retrieve top-3 chunks]              ← cosine similarity + sort
       ↓
  [build a prompt with the chunks as CONTEXT]   ← NEW (this milestone's heart)
       ↓
  [send to LLM]                        ← qwen3-vl:235b-cloud
       ↓
  [grounded answer with citations]
```

### 🧒 Why do we need the "rules" in the prompt?
Without explicit rules, the LLM defaults to its training instincts:
- It will answer from memory, not context
- It won't cite sources
- It will confabulate when context is weak

The prompt is how we **reprogram its behavior at inference time**. Every
rule is an instruction the model tries to follow.

---

## 3. The Prompt Template — Where RAG Lives or Dies

```python
PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.

CONTEXT:
{context}

RULES:
- Answer ONLY using information from the CONTEXT above.
- If the context does not contain the answer, respond EXACTLY with:
  "I don't have enough information to answer that based on the provided context."
- Cite sources in square brackets when you use a fact, e.g. [rag_overview.md].
- Be concise. 2-4 sentences.

QUESTION: {query}

ANSWER:"""
```

Every rule is deliberate:

| Rule | Why it matters |
|---|---|
| *"ONLY using information from the CONTEXT"* | Blocks training-data leakage. The bedrock of grounded RAG. |
| *"respond EXACTLY with [refusal phrase]"* | Uniform refusal string = easy to detect programmatically ("did the model refuse?"). |
| *"Cite sources in square brackets"* | Makes every claim auditable — a user can trace any fact back to a file. |
| *"Be concise. 2-4 sentences."* | Prevents verbose hallucinogenic rambling. Concise = easier to verify. |

### Context block format
```python
context_block = "\n\n".join(
    f"[Source: {h['source']}]\n{h['text']}" for h in hits
)
```
Each chunk is prefixed with `[Source: filename]` and separated by blank
lines. Two effects:
1. LLM sees filenames → can cite them
2. Chunks are visually separate → LLM treats them as distinct facts, not
   one undifferentiated blob

---

## 4. The Code

### `src/11_rag.py` — the full loop
```python
def ask(query, store):
    hits = retrieve(query, store)           # embed → top-3
    prompt = build_prompt(query, hits)      # inject chunks into template
    answer = generate(prompt)               # call Llama / Qwen via Ollama
    return hits, answer


def generate(prompt):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()
```

**That's the entire RAG system.** ~30 lines of real logic, wrapping the
retrieval we already built.

---

## 5. What I Observed

### Query 1 — Attention in transformers ⭐
> Attention in transformers works through self-attention, where each token
> in the input "looks at" every other token to determine which ones to
> focus on [transformer_architecture.md]. This is implemented using three
> learned projections (Query, Key, and Value) with the attention score
> calculated as softmax(Q · K^T / sqrt(d_k)) · V [transformer_architecture.md].
> ...

Correct formula. Correct Q/K/V naming. Three citations, all valid.

### Query 2 — Cosine similarity ⭐
> Cosine similarity measures the angle between two vectors, with a score
> of 1.0 indicating identical direction (very similar), 0.0 meaning
> orthogonal (unrelated), and -1.0 representing opposite directions
> [embeddings_basics.md]. It is the most common metric used for measuring
> vector similarity in embeddings and vector databases [vector_databases.md].

Concise, accurate ranges, cites across two sources.

### Query 3 — Chunking ⭐
> We need chunking in RAG because embedding entire documents would make
> them too coarse for effective retrieval... (typically 200-500 tokens
> with 50-100 token overlap) [rag_overview.md].

Captures the sweet spot we learned in `notes/02_chunking.md`.

### Query 4 — The France test 🇫🇷 (THE critical query)
> I don't have enough information to answer that based on the provided
> context.

**The LLM KNEW the answer (Paris is trivial training data) but refused.**
This is the line between toy RAG and real RAG. Checkmark moment.

---

## 6. What Surprised Me

1. **The refusal worked.** I half-expected the LLM to say "Paris" anyway,
   because every LLM knows this. The model honored the prompt's
   "ONLY use CONTEXT" rule even when it had better knowledge available.
   This is a strong LLM + a well-designed prompt meeting in the middle.

2. **Citations appeared naturally in the right places.** I didn't have to
   re-engineer the output format. The LLM learned "cite in square
   brackets after facts" from one example in the rules. One-shot
   instruction following.

3. **Retrieval quality limits answer quality.** For Query 3, the top-1
   chunk was about "failure modes" (not the "why chunking matters"
   section). The LLM's answer was still correct — because it synthesized
   across the top-3, not just top-1. This taught me that **top-k matters
   for synthesis tasks**, not just top-1 accuracy.

4. **Model swap changed the game.** I originally planned to use `llama3.2`
   but hadn't pulled it. I swapped to `qwen3-vl:235b-cloud` (235B params,
   cloud-hosted). A larger model follows rules more reliably —
   particularly the "refuse when context is missing" rule, which weaker
   models often ignore. **Grounding discipline scales with model
   capability.**

---

## 7. What I Got Wrong at First

| Misconception | Correction |
|---|---|
| "If I just concatenate retrieved chunks into the prompt, the LLM will figure it out." | No — you must *explicitly* instruct it to use only that context, cite, and refuse otherwise. LLMs default to using training data + the context blended. |
| "Small local models should be fine for RAG." | Often they're not, for *grounding*. They'll ignore 'use only context' rules. Bigger, instruction-tuned models (cloud or local) are more reliable. |
| "Generation failure means retrieval failure." | Not always. Sometimes retrieval is correct but the LLM ignores the context. Debug by inspecting BOTH the retrieved chunks AND the prompt that reached the model. |
| "Refusal is a failure of the system." | Refusal is a *feature* — it's the system working correctly when it should. Never punish a correct refusal. |

---

## 8. The Grounding Test — Why It's Everything

The France query is the **single most important test** a RAG system
passes or fails:

- If the LLM says "Paris" → the prompt isn't enforcing grounding. The
  LLM is using training data. Your RAG is cosmetic.
- If the LLM says "I don't have enough information..." → grounding is
  real. The retrieved context is the *only* knowledge the model will
  draw on.

This is why your CLAUDE.md's debug mindset includes:
> *"Is it chunking, embedding model, or query phrasing?"*

Because when retrieval is right and generation is still wrong, the
fault is usually in the prompt or the model's discipline — not retrieval.

---

## 9. Follow-up Questions I Still Have

1. **How do I measure grounding quantitatively?** Right now I test with 1
   out-of-scope query. A real eval should have 10+ and measure refusal rate.
2. **What if the LLM partially hallucinates** — invents a fact while also
   using real context? How do I detect that programmatically?
3. **Does prompt position matter?** Should I put rules before or after the
   context? My prompt puts CONTEXT first, then RULES. Is that optimal?
4. **Can I use system prompts for the rules instead of stuffing them into
   the user message?** Would that be more reliable with chat models?
5. **How much does model size actually matter for grounding?** I jumped
   from llama3.2 (3B) to qwen3-vl (235B). What's the smallest model that
   would reliably refuse the France query?

---

## 10. Where This Fits in the Roadmap

- **Stage:** 2 (First LLM app — embeddings → RAG)
- **Completed:**
  - ✅ Milestones 1–4 (setup → retrieval)
  - ✅ Milestone 5: end-to-end RAG with citations + refusal (this note)
- **Up next:** Path A → Path C — BM25, hybrid search, and a 20-query eval
  set to measure quality rigorously.

**Why this matters:** Today we built the smallest thing that could
legitimately be called "RAG." Everything from here is *refinement*:
better retrieval (hybrid search), better measurement (evals), better
serving (FastAPI), better observability (Langfuse). The skeleton is
done. We flesh it out next.

---

## 11. Commands I Ran

```bash
ollama pull qwen3-vl:235b-cloud   # pulled a more capable model for grounding
python src/11_rag.py               # the moment of truth
```
