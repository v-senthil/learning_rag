"""
Milestone 6 — Part B: Streamlit RAG retrieval debugger.

Interactive UI: type a query → see side-by-side rankings from vector /
BM25 / hybrid, the grounded LLM answer, and the query plotted onto a UMAP
of the chunk embeddings. Designed to let you FEEL how each piece of the
pipeline behaves.

Run:
  streamlit run src/16_streamlit_debugger.py

Then open the URL Streamlit prints (usually http://localhost:8501).
"""

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


STORE_PATH = Path(__file__).parent.parent / "vector_store_v2.json"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3-vl:235b-cloud"
QUERY_PREFIX = "search_query: "
TOP_K = 5
ALPHA = 0.5


# ---------- BM25 (same class we've been using) ----------
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


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


# ---------- scoring helpers ----------
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


# ---------- embedding + LLM ----------
def embed_query(text: str) -> np.ndarray:
    r = ollama.embeddings(model=EMBED_MODEL, prompt=QUERY_PREFIX + text)
    return np.array(r["embedding"])


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


def generate(query: str, chunk_indices: list[int], store: list[dict]) -> str:
    ctx = "\n\n".join(
        f"[Source: {store[i]['source']}]\n{store[i]['text']}"
        for i in chunk_indices
    )
    prompt = PROMPT_TEMPLATE.format(context=ctx, query=query)
    r = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip()


# ---------- cached loading (runs once per Streamlit session) ----------
@st.cache_resource
def load_everything():
    raw = json.loads(STORE_PATH.read_text())
    for r in raw:
        r["embedding"] = np.array(r["embedding"])

    bm25 = BM25FromScratch([c["text"] for c in raw])

    # Precompute UMAP on the chunks — expensive, do once
    vectors = np.array([c["embedding"] for c in raw])
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(raw) - 1),
        min_dist=0.15,
        random_state=42,
        metric="cosine",
    )
    coords = reducer.fit_transform(vectors)
    return raw, bm25, reducer, coords


# ============== STREAMLIT UI ==============
st.set_page_config(page_title="RAG Debugger", layout="wide")
st.title("🔍 RAG Retrieval Debugger")
st.caption(
    f"Embedding: `{EMBED_MODEL}`  |  LLM: `{LLM_MODEL}`  |  top_k = {TOP_K}  |  hybrid α = {ALPHA}"
)

with st.spinner("Loading corpus and computing UMAP…"):
    store, bm25, reducer, coords_2d = load_everything()

st.success(f"Loaded {len(store)} chunks from 5 source files.")

# Query input
query = st.text_input(
    "Enter a query:",
    placeholder="e.g., How does attention work in transformers?",
)

if query:
    # ---- score with all 3 methods ----
    q_vec = embed_query(query)
    vec_scores = cosine_scores(q_vec, store)
    bm_scores = bm25.get_scores(query)
    hyb_scores = ALPHA * min_max(vec_scores) + (1 - ALPHA) * min_max(bm_scores)

    vec_top = list(np.argsort(-vec_scores)[:TOP_K])
    bm_top = list(np.argsort(-bm_scores)[:TOP_K])
    hyb_top = list(np.argsort(-hyb_scores)[:TOP_K])

    # ---- LLM answer (using hybrid top-k) ----
    st.subheader("💬 LLM Answer (grounded in hybrid top-5)")
    with st.spinner("Asking the LLM…"):
        answer = generate(query, hyb_top, store)
    st.info(answer)

    # ---- retrieved chunks, 3 tabs ----
    st.subheader("📄 Retrieved Chunks")

    def render_hits(indices, scores):
        for rank, i in enumerate(indices, start=1):
            c = store[i]
            with st.expander(
                f"**#{rank} — {c['source']} (chunk {c['chunk_index']})  ·  score = {scores[i]:.4f}**",
                expanded=(rank == 1),
            ):
                st.code(c["text"], language="markdown")

    t_hyb, t_vec, t_bm = st.tabs(["⭐ Hybrid (used for the answer)", "🧠 Vector-only", "🔤 BM25-only"])
    with t_hyb:
        render_hits(hyb_top, hyb_scores)
    with t_vec:
        render_hits(vec_top, vec_scores)
    with t_bm:
        render_hits(bm_top, bm_scores)

    # ---- UMAP map with query + retrieved chunks highlighted ----
    st.subheader("🗺️ Embedding Space (UMAP 2D)")
    st.caption(
        "Dots = chunks (colored by source). Black-outlined dots = top-5 retrieved. Red star = your query."
    )

    df = pd.DataFrame({
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
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
        height=600,
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="rgba(0,0,0,0.3)")))

    # outline the retrieved chunks with a thick black circle
    retrieved_x = [coords_2d[i, 0] for i in hyb_top]
    retrieved_y = [coords_2d[i, 1] for i in hyb_top]
    retrieved_labels = [
        f"{store[i]['source']} #{store[i]['chunk_index']} (rank {rank})"
        for rank, i in enumerate(hyb_top, 1)
    ]
    fig.add_trace(go.Scatter(
        x=retrieved_x, y=retrieved_y,
        mode="markers",
        marker=dict(size=22, color="rgba(0,0,0,0)",
                    line=dict(width=3, color="black"), symbol="circle"),
        name="Top-5 retrieved",
        hovertext=retrieved_labels,
        hoverinfo="text",
    ))

    # plot the query as a red star
    q_2d = reducer.transform(q_vec.reshape(1, -1))
    fig.add_trace(go.Scatter(
        x=[q_2d[0, 0]], y=[q_2d[0, 1]],
        mode="markers+text",
        marker=dict(size=28, color="red", symbol="star",
                    line=dict(width=2, color="black")),
        text=["QUERY"], textposition="top center",
        name="Query",
        hovertext=[query],
        hoverinfo="text",
    ))

    fig.update_layout(
        xaxis_title="UMAP dim 1", yaxis_title="UMAP dim 2",
        legend=dict(orientation="h", yanchor="top", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Type a query above to see retrieval + answer + the embedding map.")
