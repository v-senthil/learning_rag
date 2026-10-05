# 02 — Chunking Strategies

**Date:** 2026-04-20
**Stage:** 2 (First LLM app — embeddings → RAG)
**Milestone:** #3 of 7 (Chunking)
**Scripts:** `src/05_why_chunk.py`, `src/06_chunking_strategies.py`, `src/07_chunk_all_docs.py`
**Output:** `vector_store.json` (28 chunks across 5 markdown files)
**Builds on:** `notes/01_embeddings_and_similarity.md`

---

## 1. What I Built / Learned

I learned why embedding whole documents is a bad idea (scores get diluted by
averaging unrelated content), explored 4 chunking strategies side-by-side
(fixed-size, paragraph, recursive, semantic), and discovered experimentally
that **fancier is not always better** — semantic chunking silently failed on
our well-structured markdown because the threshold logic couldn't detect topic
shifts in a coherent doc. I then built a real "mini vector store" by chunking
all 5 markdown files using the recursive splitter and saving 28 chunks +
768-dim embeddings to `vector_store.json`. This JSON file is the foundation
everything in Milestone 4 (retrieval) will query.

---

## 2. The Child-Level Explanation

### 🧒 Why can't I just embed the whole file?

> "If embeddings capture meaning, wouldn't embedding the whole document just
> capture ALL the meaning?"

**Honest answer:** No — because embeddings are created by **averaging** (mean-
pooling) per-token vectors. A 1000-page book's embedding is the *average* of
every topic inside it → a blurry, useless blur.

**Library analogy:** If you asked a librarian "what's this book about?" and
she said *"a little bit of math, a little bit of history, some exercises,
an appendix about formatting, and a chapter on cooking"* — that's a whole-doc
embedding. Useless for finding the math chapter when you need it.

**Solution:** Split the doc into smaller pieces where each piece is about one
focused idea. Each piece gets its OWN embedding. Now we can find the right
piece.

---

### 🔹 The Goldilocks Problem

Chunking is choosing the right size:

| Chunk size | Problem |
|---|---|
| **Whole doc** | One vector averages every topic → blurry, low retrieval scores |
| **Every word** | Words ripped from context are meaningless ("bank" alone = ???) |
| **Every sentence** | Sentences often depend on the one before them (pronoun references) |
| **Paragraph** | Usually focused, preserves context — **sweet spot** |

**Rule of thumb for modern RAG:** 200-500 tokens per chunk (roughly
~800-2000 characters), with 10-20% overlap.

---

### 🔹 The Overlap Concept

**Child:** "What if an important idea is right at the boundary between two
chunks?"

Without overlap, the idea gets sliced in half:
```
Chunk 1: "...the retrieval pipeline fetches relevant text and"
Chunk 2: "hands it to the LLM as context before generation."
```
Neither chunk alone contains the complete thought. Retrieval fails.

**With overlap** (50-100 chars shared between consecutive chunks):
```
Chunk 1: "...the retrieval pipeline fetches relevant text and hands it to the LLM..."
Chunk 2: "...and hands it to the LLM as context before generation. Finally..."
```
Any idea that spans a boundary now lives in BOTH chunks. Retrieval catches it
either way. Cheap insurance.

---

## 3. The 4 Chunking Strategies

### Strategy 1 — Fixed-Size
> Cut every N characters. Ignore structure.

```
"Embeddings turn words into numbers. Chunking means splitt|ing docs..."
                                                           ↑ cut mid-word
```
- **Pro:** 3 lines of code.
- **Con:** Slices mid-word, mid-sentence, mid-thought.
- **When:** Never in production. Fine for throwaway experiments.

---

### Strategy 2 — Paragraph-Based
> Split on `\n\n` (double newlines). Respects human-written structure.

- **Pro:** Free paragraph boundaries that humans wrote intentionally.
- **Con:** Paragraph sizes vary wildly (1 line to 500 chars). Useless for
  badly-formatted text (scraped web pages, PDFs, raw notes).
- **When:** Clean markdown/prose where the author respected paragraph breaks.

---

### Strategy 3 — Recursive Character Splitting ⭐ (the production default)
> "Try paragraphs. If too big → try lines. If too big → try sentences. If too
> big → try words. If too big → hard cut."

Takes two params:
- `target` — desired chunk size (say 400 chars)
- `overlap` — how much consecutive chunks share (say 50 chars)

And a priority list of separators, tried largest to smallest:
```python
separators = ["\n\n", "\n", ". ", " "]
```

**The algorithm:**
1. If text fits in target → return as one chunk. ✅
2. Else split on the current separator (start with `\n\n`).
3. Greedily group pieces together until adding the next would exceed target.
4. If any single piece is itself too large, **recurse** with the next
   separator down the list.
5. If we run out of separators, do a hard character cut with overlap.

**Why it's the default:** Behaves like paragraph-splitting on clean docs,
but gracefully falls back when the author wrote one giant wall of text.

---

### Strategy 4 — Semantic Chunking
> Embed each sentence. Insert a boundary where similarity *drops sharply*
> — that's where the topic shifts.

```
Sentence A: "RAG retrieves relevant documents."
Sentence B: "It feeds them to the LLM."                 ← similar (0.82)
Sentence C: "Meanwhile, pasta takes 10 minutes to cook." ← drops! (0.22) → boundary
```

- **Pro:** Chunks are topically coherent regardless of formatting.
- **Con:** Expensive (needs embeddings upfront), sensitive to threshold
  tuning, can silently fail on coherent docs (see below).
- **When:** Large messy corpora with topic diversity — AFTER evals show
  recursive isn't good enough.

---

## 4. The Recursive Algorithm In Detail

The core idea as pseudocode:

```
def split(text, separators):
    if len(text) <= target:           # small enough? keep as one chunk
        return [text]
    if not separators:                # last resort: hard cut
        return hard_cut(text, target, overlap)
    sep = separators[0]
    parts = text.split(sep)
    # greedy grouping: combine parts until we'd exceed target
    output, buffer = [], ""
    for part in parts:
        if len(buffer + sep + part) <= target:
            buffer += sep + part
        else:
            output.append(buffer)
            if part is still too big:
                output += split(part, separators[1:])   # recurse with finer separator
            else:
                buffer = part
    return output
```

**Key insight:** This algorithm **never cuts more aggressively than it has
to**. If paragraphs already fit the target, it stops there. Only when a
paragraph is itself too big does it descend to line-level, then sentence-
level, then word-level. The cuts get progressively uglier only as needed.

---

## 5. The Code

### `src/05_why_chunk.py` — prove whole-doc loses to chunks
```python
# Embed whole doc → ONE vector. Embed paragraphs → many vectors.
# Query both. Best-chunk score > whole-doc score.
whole_doc_score = cosine_similarity(query_vec, embed(full_text))
chunk_scores = [cosine_similarity(query_vec, embed(c)) for c in chunks]
best_chunk_score = max(chunk_scores)
```

### `src/06_chunking_strategies.py` — compare all 4 side-by-side
Implements `fixed_size_chunks()`, `paragraph_chunks()`, `recursive_chunks()`,
`semantic_chunks()` and runs the same query against all of them.

### `src/07_chunk_all_docs.py` — build the mini vector store
```python
for path in DATA_DIR.glob("*.md"):
    text = path.read_text()
    chunks = recursive_chunks(text, target=400, overlap=50)
    for i, chunk in enumerate(chunks):
        store.append({
            "source": path.name,
            "chunk_index": i,
            "text": chunk,
            "embedding": embed(chunk),        # 768-dim list
        })
STORE_PATH.write_text(json.dumps(store, indent=2))
```

---

## 6. What I Observed When I Ran It

### `05_why_chunk.py` — the problem stated
```
Embed WHOLE doc as 1 vector    | 0.5416
Best matching chunk            | 0.7042   ← +0.16 higher!
```
**My reaction:** "0.16 is massive in cosine-land. Whole-doc is essentially
unusable compared to chunked."

### `06_chunking_strategies.py` — comparing all 4
```
0.7042  |  Paragraph         ← winner
0.6968  |  Fixed-Size
0.6961  |  Recursive
0.5416  |  Semantic          ← catastrophic failure (1 chunk = whole doc)
```
**My reaction:** "Wait — semantic is supposed to be the *best* one, and it
scored the *worst*? And it matched the whole-doc score exactly, meaning it
didn't chunk AT ALL?"

### `07_chunk_all_docs.py` — the actual store
```
Found 5 markdown files in data/
  embeddings_basics.md                 1398 chars → 5 chunks
  python_tips.md                       1402 chars → 5 chunks
  rag_overview.md                      1663 chars → 7 chunks
  transformer_architecture.md          1637 chars → 6 chunks
  vector_databases.md                  1449 chars → 5 chunks

✅ Saved 28 chunks → vector_store.json
Chunk size: min=78, avg=268, max=398 chars
Store file size: 572.8 KB
Embedding dim: 768
```
**My reaction:** "Clean! Every chunk under the 400 target. 572 KB for 28
chunks feels bloated but that's just JSON being verbose with floats. A real
vector DB stores vectors as binary and would be ~20x smaller."

---

## 7. What Surprised Me (the "Aha" moments)

1. **Semantic chunking failed silently.** It produced ONE chunk for the
   entire document — essentially not chunking at all — and its score
   matched the whole-doc baseline *exactly*. Without running the script,
   I'd have trusted the fancier algorithm. **Evals saved me.**

2. **Fixed-size chunking performed almost as well as recursive** (0.6968 vs
   0.6961) despite its top chunk starting mid-sentence: `" documents —
   they'd be too coarse..."`. This taught me that **embeddings are robust
   to ugly text boundaries** — the meaning survives even when the form is
   broken. Which is reassuring, but also means visually nice chunks ≠
   semantically better retrieval.

3. **Paragraph won on this doc** because the markdown was well-written.
   That's a property of *this* data, not of the algorithm. On a scraped
   web page with no blank lines, paragraph chunking would produce one
   giant chunk and lose everything.

---

## 8. What I Got Wrong at First

| Misconception | Correction |
|---|---|
| "Fancier chunking (semantic) will always outperform simple ones." | Not on coherent docs. Semantic needs topic diversity to find boundaries. Choose by data, not by sophistication. |
| "The overlap shouldn't matter much — it's just repetition." | Without overlap, any idea that spans a chunk boundary is CUT. 50-char overlap is cheap insurance against catastrophic boundary losses. |
| "Perfect chunks should be clean at sentence boundaries." | Embeddings don't care about visual cleanliness. A chunk starting mid-word can still score 0.69. What matters is topical focus, not formatting. |
| "Why do we need a vector DB? Can't I just use JSON?" | JSON works for 28 chunks. At 28,000 chunks you'd be reading ~500MB of text into memory on every query. Vector DBs index smartly so lookup is O(log n). |

---

## 9. Why Semantic Chunking Failed (the deepest lesson)

Three compounding reasons:

1. **Threshold too strict.** I set `drop_threshold=0.10` below the average
   similarity. On `rag_overview.md`, every consecutive sentence stayed near
   the average because the whole doc is about one topic. No drops → no
   boundaries → one huge chunk.

2. **The doc was too coherent.** Semantic chunking needs *genuine* topic
   shifts. If every section is still about RAG, there's nothing to detect.

3. **Sentence splitter was naive.** The regex `re.split(r'(?<=[.!?])\s+',
   text)` doesn't understand markdown headers or list items. It probably
   lumped headers into neighboring sentences weirdly.

**The meta-lesson:** Sophisticated algorithms need more care, more tuning,
and more evaluation than simple ones. A working simple strategy almost
always beats a broken fancy one. This is why Stage 2 in my CLAUDE.md says
"Prompt → RAG → Fine-tune, in that order" — each step up the ladder must
justify itself against the simpler one with evals.

---

## 10. Follow-up Questions I Still Have

1. **How do I pick the right chunk size for a new dataset?** Is there a
   rule or do I need to eval multiple sizes?
2. **What's the right overlap percentage?** I used 12.5% (50/400). Is there
   research on optimal ratios?
3. **When semantic chunking DOES work, how much better does it get than
   recursive?** Worth the complexity?
4. **How do production systems chunk PDFs with tables and images?** Our
   files are clean markdown — real data is way messier.
5. **Do embedding models handle 200 chars vs 800 chars differently?** Is
   there a "best length for this specific model" we should respect?

---

## 11. Where This Fits in the Roadmap

- **Stage:** 2 (First LLM app — embeddings → RAG)
- **Milestones completed so far:**
  - ✅ Setup + sample data (Milestone 1)
  - ✅ Embeddings + similarity (Milestone 2 — `notes/01`)
  - ✅ Chunking + mini vector store (Milestone 3 — this note)
- **Up next:** Milestone 4 — Retrieval. Writing a function that takes a
  user query, embeds it, scores against all 28 chunks in our store, and
  returns top-k. This unlocks the "R" in RAG.

**Why this matters:** The 28 chunks in `vector_store.json` are DEAD data
until we can query them. Retrieval is the bridge from "I have embeddings"
to "I have a searchable knowledge base." Then Milestone 5 wires that
retrieval into an LLM and we have real RAG.

---

## 12. Commands I Ran

```bash
python src/05_why_chunk.py
python src/06_chunking_strategies.py
python src/07_chunk_all_docs.py

# Peek inside the store
python -c "import json; d=json.load(open('vector_store.json')); print(f'chunks: {len(d)}, first source: {d[0][\"source\"]}, embedding dim: {len(d[0][\"embedding\"])}')"
```
