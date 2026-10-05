# 07 — The Reusable RAG Tool (Capstone)

**Date:** 2026-04-20
**Stage:** 2 (First LLM app — embeddings → RAG)
**Milestone:** Reusable ingestion + Q&A (CLI + Streamlit UI)
**Scripts:** `src/17_ingest.py`, `src/18_ask.py`, `src/19_rag_app.py`
**New libs:** `pypdf` (PDF text extraction)
**First real use:** ingested `client channel creation use case.json` worth of docs
**Builds on:** every prior note (01 → 06)

---

## 1. What I Built / Learned

I turned the entire learning project into a **reusable tool** — not a demo.
Three entry points share the exact same pipeline under the hood:

- `src/17_ingest.py` — CLI: point it at PDFs / MD / TXT files (or a whole
  directory) and it extracts, chunks, embeds, and saves a JSON store.
- `src/18_ask.py` — CLI: interactive REPL or one-shot `--query` over any
  saved store. Hybrid retrieval + grounded LLM answer.
- `src/19_rag_app.py` — Streamlit web UI: drag-and-drop file upload, live
  progress, chat interface, save/load stores, tunable `top_k` and `α`.

The big lesson: **everything I built across 18 previous scripts
collapses into ~200 lines of production-shaped Python.** BM25 from scratch,
recursive chunker, query/document prefixes, min-max blending, sharpened
grounding prompt, citation enforcement — all of it now works on arbitrary
documents I hand it. The learning project became a *tool* I can actually
use for work. This is the moment a learning exercise became something
re-usable.

---

## 2. The Child-Level Explanation

### 🧒 Child's first question
> "Okay, we built 18 separate scripts that each taught one piece of RAG.
> How do those pieces become one *thing* I can actually hand to someone?"

**Answer:** By splitting your system into two halves with a **contract**
between them:

```
  Half 1: INGESTION
  ──────────────────
  Input:   files (PDF, MD, TXT)
  Process: extract → chunk → embed
  Output:  JSON store (the contract)

                  ↓ (the contract is just a JSON file)

  Half 2: QUERYING
  ──────────────────
  Input:   user question + JSON store
  Process: hybrid retrieve → prompt → LLM
  Output:  grounded answer + citations
```

Once you have this contract — a **well-defined JSON store format** —
you can:
- Swap ingestion tools (add Excel support? Notion export?) without
  touching querying.
- Swap querying UIs (Streamlit today, FastAPI tomorrow, Slack bot later)
  without touching ingestion.
- Build tests against the store, not the full pipeline.

**Separation of concerns** is the engineering principle. RAG is just the
use case.

### 🧒 Why have THREE entry points?
Because different users want different things:
- **CLI (17, 18)** is scriptable and composable (pipes, cron jobs, CI)
- **UI (19)** is interactive and teachable (shows non-technical users what
  the system is doing)

Same brain, different faces.

---

## 3. The JSON Store Contract

This is the backbone that holds the system together. Every component
reads/writes this exact shape:

```json
[
  {
    "source":       "paper.pdf",
    "chunk_index":  7,
    "text":         "Attention is all you need. We propose...",
    "embedding":    [0.04, -0.12, ..., 0.33]
  },
  ...
]
```

Four fields, no more. Why these four?

| Field | Why it's there |
|---|---|
| `source` | So citations can trace back to a file |
| `chunk_index` | So you can reproduce / debug which piece of the file this is |
| `text` | What the LLM actually sees in context |
| `embedding` | The 768-dim vector for cosine similarity |

**No metadata we don't use yet.** Keep the contract minimal. Production
systems add timestamps, tags, page numbers, parent doc IDs — but only
when those fields are actively queried. Don't carry dead weight.

---

## 4. The Ingestion Tool (`17_ingest.py`)

### The flow
```
  CLI arg: files or directory
       ↓
  extract_text()    ← pypdf for .pdf; read for .md/.txt
       ↓
  recursive_chunks()  ← target 400 chars, 50 overlap
       ↓
  embed_document()  ← prepend 'search_document: ', call Ollama
       ↓
  append to store
       ↓
  write JSON
```

### Key features I added beyond the basic flow
- **Deduplication by filename** — if you ingest the same file twice, it's
  skipped. Prevents accidental double-indexing.
- **Append mode (`--append`)** — merge new files into an existing store
  instead of overwriting.
- **Directory traversal (`--recursive`)** — point at a folder, get
  everything supported inside it.
- **Graceful failures** — unsupported extensions, empty files, broken PDFs
  all produce warnings, never crashes.

### One gotcha I learned
Some PDFs have NO extractable text (scanned image PDFs). `pypdf`
returns empty strings for those. My code now detects that:

```python
if not text.strip():
    st.warning(f"{uf.name}: no extractable text (image-only PDF?)")
    continue
```

**The fix for image PDFs is OCR** (Tesseract, `pdfplumber`), not pypdf.
That's out of scope for this project but worth flagging.

---

## 5. The CLI Q&A Tool (`18_ask.py`)

### Two modes

**Interactive REPL (the default):**
```bash
python src/18_ask.py --store research.json
❓ what does the paper say about scaling?
💡 [grounded answer with citations]
📄 Sources used:
   1. [hybrid 0.89 | vec 0.78 | bm25 6.12] paper.pdf #7 — ...
   ...
```

**One-shot (for scripts/automation):**
```bash
python src/18_ask.py -s research.json -q "main findings?"
```

### The full score breakdown per source
Every hit shows **three scores**, not one:
```
[hybrid 0.893 | vec 0.78 | bm25 6.12]
```
This is the debug view I missed in early milestones. Now I can see
at a glance *why* each chunk was retrieved — was it a semantic hit, a
keyword hit, or both? Makes retrieval failures diagnosable in seconds.

---

## 6. The Streamlit UI (`19_rag_app.py`)

### Design philosophy
A **chat-like interface** (using `st.chat_message` + `st.chat_input`) with
a **sidebar for config**. Mirrors how ChatGPT and Claude desktop apps
work, because users already know that pattern.

### Panels (sidebar)
- **📥 Upload** — drag-drop PDF/MD/TXT, multi-file. "Ingest" button with
  live progress bars (per-file overall + per-chunk inner).
- **📊 Current store** — live chunk count, file count, per-file breakdown,
  "save" with a name, "clear" to wipe.
- **📂 Load existing store** — dropdown of saved stores, one-click load.
- **⚙️ Retrieval settings** — `top_k` slider (1–10), `α` slider (0.0–1.0).
  Sliders update live for the NEXT query — try both to see retrieval
  shift.

### Main area
- **Empty state**: brief pipeline explanation (for onboarding non-technical users)
- **After ingestion**: a chat. Q and A bubbles. Every answer has an
  expandable "📄 Sources used" block with full chunk text + all 3 scores.

### The one Streamlit trick that made this easy
```python
@st.cache_resource
def load_everything():
    # loads store, builds BM25, runs UMAP — expensive
    ...
```
Without `cache_resource`, Streamlit re-runs the ENTIRE file on every
user interaction. This would re-build BM25, re-run UMAP, re-load JSON…
nightmare latency. `cache_resource` caches computed objects in memory
until the app restarts. Fix learned the hard way in earlier scripts.

### Session state
```python
if "store" not in st.session_state:
    st.session_state.store = []
if "chat" not in st.session_state:
    st.session_state.chat = []
```
`st.session_state` is how Streamlit persists data across interactions
(uploads, button clicks) without re-running everything. Chat history
lives here. Without it, every click would wipe conversation history.

---

## 7. What I Observed (First Real-World Use)

I ingested a real Zoho doc (`client channel creation use case.json`-
related content) to test beyond the practice markdown files. Observations:

1. **The UI made debugging visual.** Early on, one query retrieved the
   wrong section. Opening "Sources used" immediately showed me which
   chunk won and with what scores — I could see the BM25 keyword match
   was dominating incorrectly. 5 seconds of debugging vs. 10 minutes of
   CLI print-wrangling.

2. **PDF extraction is lossy.** Tables, code blocks, and formatted lists
   come out as raw text with no structure preserved. Chunks still work
   for Q&A but sometimes lose context (a code block's language, a
   table's column headers). Worth remembering when ingesting
   technical docs.

3. **Chunking target=400 chars is too small for dense PDFs.** Academic
   papers and dense technical docs benefit from target=600–800. The UI
   doesn't expose this as a slider yet — good next improvement.

4. **Hybrid still wins on real data.** BM25 correctly prioritizes chunks
   containing specific terms in the query (e.g., a function or API
   name). Vectors correctly handle paraphrased questions. Same pattern
   as my eval set — the tradeoff doesn't disappear just because the
   corpus is bigger.

---

## 8. What Surprised Me

1. **Everything composable.** The same `recursive_chunks`,
   `BM25FromScratch`, `cosine_scores`, `min_max`, `embed_document`,
   `generate_answer` functions now appear in 3 different entry points
   unchanged. The only thing that varies is the *shell around them*
   (CLI args vs Streamlit widgets). **When your core logic is clean,
   packaging it in different UIs is mechanical.**

2. **The UI changed what I could test.** With the Streamlit interface, I
   started asking my documents *weird* questions — intentionally
   adversarial, paraphrased, partially off-topic. CLI testing was
   always "does this exact query work?" UI testing was "does this feel
   right?" Very different depth of exploration.

3. **Save/Load stores ended the friction of iteration.** Before this,
   every time I wanted to test a query, I re-ran ingestion. Now I
   ingest once, save, and every future session loads instantly. A
   2-line feature that fundamentally changes how I use the system.

4. **File uploader + progress bars sold the experience.** Watching
   chunks get embedded live ("embedding 15/47…") makes the abstract
   work feel real. For myself, for anyone I demo this to, this is
   what makes RAG click.

---

## 9. What I Got Wrong at First

| Misconception | Correction |
|---|---|
| "I should build one big `app.py` with everything." | Splitting ingestion from querying is the key structural choice. They have different failure modes, different latencies, different UIs. Keep them separate. |
| "I'll rewrite the BM25 / embedding / retrieval code for each entry point." | Copy-paste is fine for a learning project. In a real team, move shared logic to a `core.py` module. Same principle either way: DRY. |
| "File uploads in Streamlit are complicated." | They're dead simple — `st.file_uploader(..., accept_multiple_files=True)` returns a list. The complexity is in *handling* different file types, not in the UI. |
| "The JSON store will slow down as it grows." | It does — read-all-into-memory doesn't scale. But it works fine up to ~5-10k chunks. Swap to Chroma/FAISS when you hit a real wall, not before. |

---

## 10. The Full Tech Stack (final)

| Layer | Tech | Why |
|---|---|---|
| Embedding model | `nomic-embed-text` via Ollama (local) | 768-dim, fast, free, prefix-aware |
| LLM | `qwen3-vl:235b-cloud` via Ollama | Large enough for reliable grounding |
| Vector math | NumPy | Cosine similarity by hand |
| Keyword search | Custom BM25 from scratch | 30 lines, fully understood |
| Chunking | Custom recursive splitter | Paragraph → line → sentence → word |
| PDF extraction | `pypdf` | Pure Python, no system deps |
| Storage | Plain JSON files | Inspectable, portable, diff-friendly |
| CLI | Python stdlib `argparse` | No frameworks needed |
| Web UI | `streamlit` | Pythonic, no frontend work |
| Visualization | `umap-learn` + `plotly` | Interactive 2D embedding maps |

**Zero heavy frameworks.** No LangChain, no LlamaIndex, no vector DB.
Everything small, everything understood, everything debuggable.

---

## 11. Follow-up Questions I Still Have

1. **When is a real vector DB necessary?** My sense: ~10k chunks is the
   JSON ceiling. At 100k+ chunks Chroma/FAISS becomes essential. What's
   the exact threshold on my hardware?

2. **How do I add structured metadata (page numbers, doc type,
   timestamps) without breaking the minimal contract?** Maybe a
   `"meta"` dict field that's optional and ignored by default?

3. **Can I combine multiple stores at query time?** E.g., "search my
   notes AND my papers stores simultaneously" — would need score
   normalization across stores.

4. **What's the right way to handle document updates?** If a file
   changes, I currently need to delete the old chunks and re-ingest.
   Production systems version + diff. How would that work here?

5. **Could I stream LLM responses for lower latency feel?** Ollama
   supports streaming. Streamlit has `st.write_stream`. Haven't wired
   them together.

6. **How much would a reranker improve results on real data?** My evals
   are done on practice docs. Real documents have messier language —
   the gap might be bigger.

---

## 12. Where This Fits in the Roadmap

- **Stage:** 2 (First LLM app — embeddings → RAG)
- **Status:** 🎉 **This closes the core RAG curriculum.**
- **Completed milestones:**
  - ✅ Milestone 1: Setup + sample data
  - ✅ Milestone 2: Embeddings + similarity (`notes/01`)
  - ✅ Milestone 3: Chunking (`notes/02`)
  - ✅ Milestone 4: Retrieval + prefixes (`notes/03`)
  - ✅ Milestone 5: End-to-end RAG (`notes/04`)
  - ✅ Path A: Hybrid search (`notes/05`)
  - ✅ Path C: Evals + tuning (`notes/06`)
  - ✅ Milestone 6: Visualization (UMAP + Streamlit debugger)
  - ✅ **Milestone 7: Reusable tool (this note)** ← capstone

- **Optional remaining (Stage 2 polish):**
  - [ ] Reranker (cross-encoder) — address PARA-01 failure
  - [ ] FastAPI wrapper for HTTP API
  - [ ] Langfuse or Phoenix instrumentation
  - [ ] Public deployment (Docker / Fly / Render)

- **Next major step:** Stage 3 — Agents. This means tool use, planning
  loops, MCP servers, multi-step reasoning. Different mental model from
  RAG (agents *act*, RAG just *retrieves*). The foundation we built here
  carries forward: every agent that uses documents will still need
  these retrieval primitives.

**Why this matters:** Anyone can follow a RAG tutorial. Very few people
follow it *end-to-end*, measure it with evals, tune the prompt, build
the UI, and turn it into a reusable tool. That complete arc is what
separates "I did a tutorial" from "I can build this for production."
You now have artifacts you can show, use, and point others to.

---

## 13. Commands I Ran

```bash
# install new deps
pip install -r requirements.txt   # now includes pypdf, rank-bm25

# CLI ingestion
python src/17_ingest.py data/                              # ingest folder
python src/17_ingest.py paper.pdf --output research.json   # single file
python src/17_ingest.py new.pdf --output research.json --append

# CLI Q&A
python src/18_ask.py                                       # interactive
python src/18_ask.py -s research.json -q "what is X?"      # one-shot

# Full Streamlit UI
streamlit run src/19_rag_app.py                            # the main tool
```

---

## 14. The Meta-Lesson

If I compare where I started (note 01: running my first embedding)
with where I am now (note 07: a reusable tool that ingests PDFs and
answers questions with grounded citations), the technical distance is
enormous. But every step was small. The arc worked because:

1. **I built before abstracting.** I had 4 chunking strategies in code
   before I understood which to pick. The intuition came from *running*
   things, not reading about them.

2. **I measured before tuning.** Evals came before I tweaked the prompt
   or the `top_k`. Every optimization was informed, not guessed.

3. **I wrote notes during, not after.** Every "aha" is frozen in a
   `notes/` markdown, and cross-referenced to the others. The notes
   folder is my actual second brain for this material.

4. **I refused to use magic.** I wrote BM25 from scratch. I wrote
   min-max normalization from scratch. I wrote the recursive splitter
   from scratch. When something broke, I knew exactly where to look
   because I'd written every line.

**This is how you learn a technical field.** Not by reading. Not by
watching. By building, measuring, wrestling, and documenting — in
that order, every time.
