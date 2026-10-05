"""
Milestone 7 — Full RAG UI (Streamlit).

A single web app that wraps every technique from the learning project:
  • Upload PDF / MD / TXT files (multi-file supported)
  • Recursive chunking with overlap
  • Embedding with nomic-embed-text 'search_document:' prefix
  • Hybrid retrieval (cosine + BM25, min-max blended, tunable α)
  • Grounded LLM answer with citations + refusal discipline
  • Persistent session: chat history, source breakdown, save/load
  • AI Input (full prompt) + AI Output (answer) both inspectable
  • UMAP map showing where each query lands in the embedding space

Run:
  streamlit run src/19_rag_app.py
"""

import io
import json
import math
import re
from collections import Counter
from pathlib import Path
import numpy as np
import ollama
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import umap
from pypdf import PdfReader


# ================= config =================
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3-vl:235b-cloud"
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "
DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.5
STORE_DIR = Path(__file__).parent.parent / "stores"
STORE_DIR.mkdir(exist_ok=True)


# ================= chunking =================
def recursive_chunks(text: str, target: int = 400, overlap: int = 50) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]

    def split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= target:
            return [t] if t.strip() else []
        if not seps:
            return [t[i:i + target] for i in range(0, len(t), target - overlap)]
        sep = seps[0]
        parts = t.split(sep)
        out, buffer = [], ""
        for part in parts:
            candidate = buffer + (sep if buffer else "") + part
            if len(candidate) <= target:
                buffer = candidate
            else:
                if buffer:
                    out.append(buffer)
                if len(part) > target:
                    out.extend(split(part, seps[1:]))
                    buffer = ""
                else:
                    buffer = part
        if buffer:
            out.append(buffer)
        return [c for c in out if c.strip()]

    return split(text, separators)


# ================= BM25 =================
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


# ================= embedding + retrieval =================
def embed_document(text: str) -> list[float]:
    model = st.session_state.get("embed_model", EMBED_MODEL)
    r = ollama.embeddings(model=model, prompt=DOC_PREFIX + text)
    return r["embedding"]


def embed_query(text: str) -> np.ndarray:
    model = st.session_state.get("embed_model", EMBED_MODEL)
    r = ollama.embeddings(model=model, prompt=QUERY_PREFIX + text)
    return np.array(r["embedding"])


# ================= Ollama model discovery =================
def fetch_ollama_models() -> list[str] | None:
    """Ask Ollama for every installed (and registered cloud) model.
    Returns None if Ollama is unreachable — the UI uses that to show an error."""
    try:
        result = ollama.list()
        names = [m.get("name") or m.get("model") for m in result.get("models", [])]
        return sorted({n for n in names if n})
    except Exception:
        return None


def classify_models(models: list[str]) -> tuple[list[str], list[str]]:
    """Split into (llm_candidates, embed_candidates) using a name heuristic.
    Anything with 'embed' in its name → embedding model. Everything else → LLM."""
    embed = [m for m in models if "embed" in m.lower()]
    llm = [m for m in models if "embed" not in m.lower()]
    return llm, embed


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


def hybrid_retrieve(q_vec: np.ndarray, query: str, store: list[dict],
                    bm25: BM25FromScratch, top_k: int, alpha: float) -> list[dict]:
    """Return top-k hits with store index, hybrid score, vec score, bm25 score."""
    vec = cosine_scores(q_vec, store)
    bm = bm25.get_scores(query)
    hybrid = alpha * min_max(vec) + (1 - alpha) * min_max(bm)
    top_idx = np.argsort(-hybrid)[:top_k]
    return [{
        "store_index": int(i),
        "score": float(hybrid[i]),
        "vec_score": float(vec[i]),
        "bm25_score": float(bm[i]),
        "source": store[i]["source"],
        "chunk_index": store[i]["chunk_index"],
        "text": store[i]["text"],
    } for i in top_idx]


# ================= prompt + generation =================
PROMPT_TEMPLATE = """You are a helpful assistant answering questions using the context below.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question using the CONTEXT. You MAY synthesize, infer, and combine facts across chunks — the exact wording of the question does not need to appear in the context.
- Do NOT introduce facts that are not supported by the context.
- ADJACENT facts are NOT sufficient grounds to answer. If the context mentions related concepts but does not cover the specific task, entity, or concept asked about, REFUSE.
- If you refuse, respond EXACTLY with:
  "I don't have enough information to answer that based on the provided context."
- Cite sources in square brackets when you use a fact, e.g. [paper.pdf].
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


def generate_answer(prompt: str) -> str:
    model = st.session_state.get("llm_model", LLM_MODEL)
    r = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip()


# ================= file extraction =================
def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    elif ext in {".md", ".txt"}:
        return uploaded_file.read().decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported extension: {ext}")


# ================= UMAP (recomputed when store changes) =================
def rebuild_umap():
    """Compute UMAP reducer + 2D coords from the current store. Stores both
    in session_state so query points can be .transform()'d later."""
    store = st.session_state.store
    if len(store) < 3:
        st.session_state.umap_reducer = None
        st.session_state.umap_coords = None
        return
    vectors = np.array([c["embedding"] for c in store])
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(store) - 1),
        min_dist=0.15,
        random_state=42,
        metric="cosine",
    )
    coords = reducer.fit_transform(vectors)
    st.session_state.umap_reducer = reducer
    st.session_state.umap_coords = coords


# ================= ingestion =================
def ingest_uploads(uploaded_files) -> int:
    existing = {c["source"] for c in st.session_state.store}
    total_new = 0

    overall = st.progress(0.0, text="Starting ingestion...")
    for fi, uf in enumerate(uploaded_files):
        if uf.name in existing:
            st.warning(f"⏭️  {uf.name} already in store — skipped")
            continue
        try:
            text = extract_text_from_upload(uf)
        except Exception as e:
            st.error(f"❌ {uf.name}: extraction failed — {e}")
            continue

        if not text.strip():
            st.warning(f"⚠️  {uf.name}: no extractable text (image-only PDF?)")
            continue

        chunks = recursive_chunks(text)
        st.info(f"📄 {uf.name}: {len(text):,} chars → {len(chunks)} chunks. Embedding...")

        inner = st.progress(0.0, text=f"Embedding {uf.name}")
        for i, chunk in enumerate(chunks):
            vec = embed_document(chunk)
            st.session_state.store.append({
                "source": uf.name,
                "chunk_index": i,
                "text": chunk,
                "embedding": np.array(vec),
            })
            inner.progress((i + 1) / len(chunks), text=f"Embedding {uf.name}: {i+1}/{len(chunks)}")
        inner.empty()
        total_new += len(chunks)
        overall.progress((fi + 1) / len(uploaded_files), text=f"Done {fi+1}/{len(uploaded_files)} files")

    overall.empty()

    if st.session_state.store:
        st.session_state.bm25 = BM25FromScratch([c["text"] for c in st.session_state.store])
        with st.spinner("Computing UMAP map…"):
            rebuild_umap()

    return total_new


# ================= persistence =================
def save_store(name: str):
    path = STORE_DIR / f"{name}.json"
    data = [{
        "source": c["source"],
        "chunk_index": c["chunk_index"],
        "text": c["text"],
        "embedding": c["embedding"].tolist(),
    } for c in st.session_state.store]
    path.write_text(json.dumps(data, indent=2))
    return path


def load_store(name: str):
    path = STORE_DIR / f"{name}.json"
    data = json.loads(path.read_text())
    st.session_state.store = [{
        "source": r["source"],
        "chunk_index": r["chunk_index"],
        "text": r["text"],
        "embedding": np.array(r["embedding"]),
    } for r in data]
    st.session_state.bm25 = BM25FromScratch([c["text"] for c in st.session_state.store])
    st.session_state.chat = []
    rebuild_umap()


def list_stores() -> list[str]:
    return sorted([p.stem for p in STORE_DIR.glob("*.json")])


# ================= Streamlit UI =================
st.set_page_config(page_title="RAG Tool", layout="wide", page_icon="📚")

# init session state
for key, default in [
    ("store", []),
    ("bm25", None),
    ("chat", []),
    ("umap_reducer", None),
    ("umap_coords", None),
    ("embed_model", EMBED_MODEL),
    ("llm_model", LLM_MODEL),
    ("available_models", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ----- Sidebar -----
with st.sidebar:
    st.title("📚 RAG Tool")

    # ---- Model picker (backed by Ollama's list API) ----
    # Auto-fetch once; the user can refresh manually anytime.
    if st.session_state.available_models is None:
        st.session_state.available_models = fetch_ollama_models()

    with st.expander(
        f"🤖 Models — LLM: `{st.session_state.llm_model}`  ·  Embed: `{st.session_state.embed_model}`",
        expanded=False,
    ):
        if st.button("🔄 Refresh model list", use_container_width=True):
            with st.spinner("Fetching from Ollama…"):
                st.session_state.available_models = fetch_ollama_models()

        models = st.session_state.available_models
        if models is None:
            st.error("Ollama unreachable. Is `ollama serve` running?")
        elif not models:
            st.warning("No models found. Try `ollama pull <name>` first.")
        else:
            llm_candidates, embed_candidates = classify_models(models)

            # ensure the currently-selected models are always in their dropdowns
            # (cloud models sometimes don't appear in list(); preserve them anyway).
            if st.session_state.llm_model not in llm_candidates:
                llm_candidates = [st.session_state.llm_model] + llm_candidates
            if st.session_state.embed_model not in embed_candidates:
                embed_candidates = [st.session_state.embed_model] + embed_candidates

            # --- LLM: safe to switch anytime ---
            new_llm = st.selectbox(
                "LLM (for generation)",
                options=llm_candidates,
                index=llm_candidates.index(st.session_state.llm_model),
                help="Changes apply on the NEXT query. Past answers keep their original model.",
            )
            if new_llm != st.session_state.llm_model:
                st.session_state.llm_model = new_llm
                st.success(f"LLM → {new_llm}")

            # --- Embedding: locked when store has data (different models = incompatible vectors) ---
            store_has_data = len(st.session_state.store) > 0
            new_embed = st.selectbox(
                "Embedding model (for ingestion + query)",
                options=embed_candidates,
                index=embed_candidates.index(st.session_state.embed_model),
                disabled=store_has_data,
                help=(
                    "Switching embedding models invalidates all stored vectors. "
                    "Clear the store first to change this."
                ),
            )
            if not store_has_data and new_embed != st.session_state.embed_model:
                st.session_state.embed_model = new_embed
                st.success(f"Embedding model → {new_embed}")

            if store_has_data:
                st.caption("🔒 Embedding model is locked — clear the store below to change it.")

            st.caption(f"{len(models)} model(s) available in Ollama.")

    st.divider()

    # --- Upload ---
    st.subheader("📥 Upload documents")
    uploaded = st.file_uploader(
        "Drop PDF / Markdown / text files here",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("🚀 Ingest", type="primary", use_container_width=True):
        with st.status("Ingesting…", expanded=True) as status:
            new_count = ingest_uploads(uploaded)
            status.update(label=f"✅ Added {new_count} new chunks", state="complete")

    st.divider()

    # --- Current store ---
    if st.session_state.store:
        st.subheader("📊 Current store")
        sources = {c["source"] for c in st.session_state.store}
        col1, col2 = st.columns(2)
        col1.metric("Chunks", len(st.session_state.store))
        col2.metric("Files", len(sources))

        with st.expander("Files in store"):
            for s in sorted(sources):
                count = sum(1 for c in st.session_state.store if c["source"] == s)
                st.text(f"• {s}  ({count})")

        save_name = st.text_input("Save as…", placeholder="my_kb", key="save_name")
        if st.button("💾 Save store", use_container_width=True) and save_name:
            path = save_store(save_name.strip())
            st.success(f"Saved to {path.name}")

        if st.button("🗑️  Clear store", use_container_width=True):
            st.session_state.store = []
            st.session_state.bm25 = None
            st.session_state.chat = []
            st.session_state.umap_reducer = None
            st.session_state.umap_coords = None
            st.rerun()

    st.divider()

    # --- Load existing store ---
    st.subheader("📂 Load existing store")
    saved = list_stores()
    if saved:
        choice = st.selectbox("Pick one", [""] + saved, index=0, label_visibility="collapsed")
        if choice and st.button("Load", use_container_width=True):
            with st.spinner(f"Loading '{choice}' and computing UMAP…"):
                load_store(choice)
            st.success(f"Loaded '{choice}'")
            st.rerun()
    else:
        st.caption("No saved stores yet.")

    st.divider()

    # --- Settings ---
    with st.expander("⚙️ Retrieval settings"):
        top_k = st.slider("top_k", 1, 10, DEFAULT_TOP_K,
                          help="Number of chunks sent to the LLM.")
        alpha = st.slider("α (hybrid weight)", 0.0, 1.0, DEFAULT_ALPHA, 0.1,
                          help="1.0 = vector only. 0.0 = BM25 only. 0.5 = balanced.")


# ----- Empty state -----
if not st.session_state.store:
    st.title("📚 RAG Tool — upload to start")
    st.markdown(
        """
        **What this app does:**
        1. You **upload** PDF / Markdown / text files in the sidebar
        2. The app **extracts text**, splits it into **chunks with overlap**, and
           builds **768-dim embeddings** using `nomic-embed-text`
        3. When you ask a question, it runs **hybrid retrieval** (semantic vectors
           + BM25 keywords) to find the most relevant chunks
        4. The LLM answers using **only** those chunks, cites sources, and
           refuses if the docs don't cover your question

        Each answer expands to show:
          • 🧠 **AI Input** — the full prompt sent to the LLM
          • 💡 **AI Output** — the grounded answer (displayed inline)
          • 📄 **Sources used** — top-k chunks with hybrid/vec/bm25 scores
          • 🗺️ **Embedding Map** — where your query landed in UMAP 2D space

        👉 Upload files using the sidebar to get started.
        """
    )
    st.stop()


# ----- Chat interface -----
st.title("💬 Ask your documents")
st.caption(
    f"{len(st.session_state.store)} chunks from "
    f"{len({c['source'] for c in st.session_state.store})} file(s)  ·  "
    f"top_k={top_k}  ·  α={alpha}  ·  "
    f"LLM: `{st.session_state.llm_model}`  ·  "
    f"Embed: `{st.session_state.embed_model}`"
)


# ================= rendering helpers =================
def render_ai_input(prompt: str):
    with st.expander("🧠 AI Input (full prompt sent to the LLM)", expanded=False):
        st.code(prompt, language="text")
        st.caption(f"Prompt length: {len(prompt):,} characters")


def render_sources(hits: list[dict]):
    with st.expander(f"📄 Sources used ({len(hits)} chunks)", expanded=False):
        for rank, h in enumerate(hits, 1):
            st.markdown(
                f"**#{rank} — {h['source']} · chunk {h['chunk_index']}**  "
                f"<span style='color:gray'>hybrid: {h['score']:.3f} · "
                f"vec: {h['vec_score']:.3f} · bm25: {h['bm25_score']:.2f}</span>",
                unsafe_allow_html=True,
            )
            st.code(h["text"], language="markdown")
            if rank < len(hits):
                st.divider()


def render_umap(query: str, hits: list[dict], q_vec: np.ndarray):
    with st.expander("🗺️ Embedding Map (UMAP 2D — where your query landed)", expanded=False):
        if st.session_state.umap_reducer is None:
            st.info("Need at least 3 chunks in the store to render the map.")
            return

        coords = st.session_state.umap_coords
        store = st.session_state.store
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "source": [c["source"] for c in store],
            "label": [f"{c['source']} #{c['chunk_index']}" for c in store],
            "preview": [
                c["text"][:120].replace("\n", " ") + ("..." if len(c["text"]) > 120 else "")
                for c in store
            ],
        })

        fig = px.scatter(
            df, x="x", y="y", color="source",
            hover_data={"label": True, "preview": True, "x": False, "y": False},
            height=550,
        )
        fig.update_traces(marker=dict(size=11, line=dict(width=1, color="rgba(0,0,0,0.3)")))

        # outline retrieved chunks
        retrieved_idx = [h["store_index"] for h in hits]
        fig.add_trace(go.Scatter(
            x=[coords[i, 0] for i in retrieved_idx],
            y=[coords[i, 1] for i in retrieved_idx],
            mode="markers",
            marker=dict(size=22, color="rgba(0,0,0,0)",
                        line=dict(width=3, color="black"), symbol="circle"),
            name="Top-k retrieved",
            hovertext=[
                f"Rank {rank}: {store[i]['source']} #{store[i]['chunk_index']}"
                for rank, i in enumerate(retrieved_idx, 1)
            ],
            hoverinfo="text",
        ))

        # plot the query as a red star
        try:
            q_2d = st.session_state.umap_reducer.transform(q_vec.reshape(1, -1))
            fig.add_trace(go.Scatter(
                x=[q_2d[0, 0]], y=[q_2d[0, 1]],
                mode="markers+text",
                marker=dict(size=26, color="red", symbol="star",
                            line=dict(width=2, color="black")),
                text=["QUERY"], textposition="top center",
                name="Query",
                hovertext=[query],
                hoverinfo="text",
            ))
        except Exception as e:
            st.warning(f"Could not project query onto map: {e}")

        fig.update_layout(
            xaxis_title="UMAP dim 1", yaxis_title="UMAP dim 2",
            legend=dict(orientation="h", yanchor="top", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Dots = chunks (colored by source). Black-outlined dots = retrieved top-k. "
            "Red star = your query. **Note:** UMAP is a lossy 2D view of 768-dim space — "
            "use scores for ground truth, not visual distance."
        )


def render_turn(turn: dict):
    """Render one full Q&A turn: AI input + output + sources + UMAP."""
    render_ai_input(turn["prompt"])
    render_sources(turn["hits"])
    render_umap(turn["query"], turn["hits"], turn["q_vec"])


# ----- Replay history -----
for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.markdown("**💡 AI Output**")
        st.write(turn["answer"])
        render_turn(turn)


# ----- New query -----
query = st.chat_input("Ask a question about your documents…")
if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Embedding query & retrieving chunks…"):
            q_vec = embed_query(query)
            hits = hybrid_retrieve(
                q_vec, query,
                st.session_state.store,
                st.session_state.bm25,
                top_k=top_k, alpha=alpha,
            )
            prompt = build_prompt(query, hits)

        with st.spinner("Asking the LLM…"):
            answer = generate_answer(prompt)

        st.markdown("**💡 AI Output**")
        st.write(answer)
        render_ai_input(prompt)
        render_sources(hits)
        render_umap(query, hits, q_vec)

    st.session_state.chat.append({
        "query": query,
        "answer": answer,
        "hits": hits,
        "prompt": prompt,
        "q_vec": q_vec,
    })
