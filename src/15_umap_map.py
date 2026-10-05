"""
Milestone 6 — Part A: UMAP projection of the embedding space.

Takes all 28 chunks (each a 768-dim vector), projects them to 2D with UMAP,
and saves an interactive HTML scatter. Each dot is a chunk, colored by
source file. Hover reveals the chunk preview. Optionally plots a query
embedding as a red star in the same space.

Run:
  python src/15_umap_map.py                             # just chunks
  python src/15_umap_map.py --query "what is RAG?"      # + query as red star
Open: rag_project/embedding_map.html
"""

import argparse
import json
from pathlib import Path
import numpy as np
import ollama
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap


STORE_PATH = Path(__file__).parent.parent / "vector_store_v2.json"
OUTPUT_HTML = Path(__file__).parent.parent / "embedding_map.html"
EMBED_MODEL = "nomic-embed-text"
QUERY_PREFIX = "search_query: "


def load_store() -> list[dict]:
    return json.loads(STORE_PATH.read_text())


def embed_query(text: str) -> np.ndarray:
    """Embed a query using the same prefix convention as the store."""
    r = ollama.embeddings(model=EMBED_MODEL, prompt=QUERY_PREFIX + text)
    return np.array(r["embedding"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, default=None,
                   help="optional query — projected onto the same UMAP as a red star")
    args = p.parse_args()

    store = load_store()
    n_chunks = len(store)
    print(f"Loaded {n_chunks} chunks from {STORE_PATH.name}")

    # stack chunk embeddings into one matrix: (28, 768)
    chunk_vectors = np.array([c["embedding"] for c in store])
    print(f"Chunk matrix shape: {chunk_vectors.shape}")

    # UMAP with cosine metric (match our retrieval metric — important!)
    # n_neighbors small since we have only 28 points
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, n_chunks - 1),
        min_dist=0.15,
        random_state=42,
        metric="cosine",
    )
    print("Running UMAP projection (this may take ~5s)...")
    coords_2d = reducer.fit_transform(chunk_vectors)
    print(f"Projected shape: {coords_2d.shape}")

    # build dataframe for plotly
    df = pd.DataFrame({
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
        "source": [c["source"] for c in store],
        "chunk_index": [c["chunk_index"] for c in store],
        "preview": [
            c["text"][:120].replace("\n", " ") + ("..." if len(c["text"]) > 120 else "")
            for c in store
        ],
    })
    df["label"] = df["source"] + " #" + df["chunk_index"].astype(str)

    fig = px.scatter(
        df,
        x="x", y="y",
        color="source",
        hover_data={"label": True, "preview": True, "x": False, "y": False},
        title=f"Embedding space — UMAP 2D ({n_chunks} chunks from 5 markdown files)",
        width=1100,
        height=700,
    )
    fig.update_traces(marker=dict(size=13, line=dict(width=1, color="rgba(0,0,0,0.3)")))
    fig.update_layout(
        legend=dict(title="Source file", orientation="v"),
        xaxis_title="UMAP dim 1",
        yaxis_title="UMAP dim 2",
    )

    # optionally project a query into the same space
    if args.query:
        print(f"Embedding query: '{args.query}'")
        q_vec = embed_query(args.query).reshape(1, -1)
        q_2d = reducer.transform(q_vec)
        fig.add_trace(go.Scatter(
            x=[q_2d[0, 0]], y=[q_2d[0, 1]],
            mode="markers+text",
            marker=dict(size=28, color="red", symbol="star",
                        line=dict(width=2, color="black")),
            text=["QUERY"],
            textposition="top center",
            name="Query",
            hovertext=[args.query],
            hoverinfo="text",
        ))

    fig.write_html(str(OUTPUT_HTML))
    print(f"\n✅ Saved {OUTPUT_HTML.name}")
    print(f"   Open with:  open {OUTPUT_HTML}")
    print(f"   Or drag the file into any browser.")


if __name__ == "__main__":
    main()
