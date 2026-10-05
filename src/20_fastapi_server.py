"""
FastAPI server for the RAG tool — backend for the world-class UI.

Endpoints:
  GET    /                            — serves the web app (static/index.html)
  GET    /health                      — ping + store size
  GET    /models                      — {llm: [...], embed: [...], current: {...}}
  POST   /models/llm                  — switch LLM
  POST   /models/embed                — switch embedding model (locked if store non-empty)
  GET    /stores                      — list saved stores
  POST   /stores/{name}/load          — load a saved store
  POST   /stores/{name}/save          — persist current store
  DELETE /store                       — clear in-memory store
  POST   /ingest                      — multipart upload → chunk + embed
  POST   /ask                         — JSON query → answer + sources
  POST   /ask/stream                  — NDJSON streaming response
  GET    /docs                        — Swagger UI (auto-generated)

Run:
  python src/20_fastapi_server.py
"""

import io
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
import ollama
import umap
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader


# ================= config =================
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "qwen3-vl:235b-cloud"
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "
DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.5
BASE_DIR = Path(__file__).parent.parent
STORE_DIR = BASE_DIR / "stores"
STATIC_DIR = BASE_DIR / "static"
STORE_DIR.mkdir(exist_ok=True)


# ================= core logic =================
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


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if ext in {".md", ".txt"}:
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported extension: {ext}")


# ================= state =================
class State:
    def __init__(self):
        self.store: list[dict] = []
        self.bm25: Optional[BM25FromScratch] = None
        self.llm_model: str = DEFAULT_LLM_MODEL
        self.embed_model: str = DEFAULT_EMBED_MODEL
        self.umap_reducer = None
        self.umap_coords: Optional[np.ndarray] = None

    def rebuild_bm25(self):
        self.bm25 = BM25FromScratch([c["text"] for c in self.store]) if self.store else None

    def rebuild_umap(self):
        """Fit UMAP on current store. Called after ingest/load — expensive, ~seconds."""
        if len(self.store) < 3:
            self.umap_reducer = None
            self.umap_coords = None
            return
        vectors = np.array([c["embedding"] for c in self.store])
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(self.store) - 1),
            min_dist=0.15,
            random_state=42,
            metric="cosine",
        )
        self.umap_coords = reducer.fit_transform(vectors)
        self.umap_reducer = reducer

    def clear(self):
        self.store = []
        self.bm25 = None
        self.umap_reducer = None
        self.umap_coords = None


state = State()


def embed_document(text: str) -> list[float]:
    r = ollama.embeddings(model=state.embed_model, prompt=DOC_PREFIX + text)
    return r["embedding"]


def embed_query(text: str) -> np.ndarray:
    r = ollama.embeddings(model=state.embed_model, prompt=QUERY_PREFIX + text)
    return np.array(r["embedding"])


def fetch_ollama_models() -> list[str]:
    try:
        result = ollama.list()
        names = [m.get("name") or m.get("model") for m in result.get("models", [])]
        return sorted({n for n in names if n})
    except Exception:
        return []


def classify_models(models: list[str]) -> tuple[list[str], list[str]]:
    embed = [m for m in models if "embed" in m.lower()]
    llm = [m for m in models if "embed" not in m.lower()]
    return llm, embed


# ================= Pydantic schemas =================
class HealthResponse(BaseModel):
    status: str
    chunks: int
    files: int
    store_loaded: bool


class ModelsResponse(BaseModel):
    llm: list[str]
    embed: list[str]
    current_llm: str
    current_embed: str


class SwitchModelRequest(BaseModel):
    model: str


class AskRequest(BaseModel):
    query: str
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    alpha: float = Field(DEFAULT_ALPHA, ge=0.0, le=1.0)


class SourceHit(BaseModel):
    score: float
    vec_score: float
    bm25_score: float
    source: str
    chunk_index: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceHit]
    prompt: str


class IngestResponse(BaseModel):
    new_chunks: int
    total_chunks: int
    files_ingested: list[str]
    files_skipped: list[str]


class FileInfo(BaseModel):
    name: str
    chunks: int


class StoreDetail(BaseModel):
    total_chunks: int
    files: list[FileInfo]


# ================= FastAPI =================
app = FastAPI(title="RAG Tool API", version="1.1.0")


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "UI not found. Place files in static/ or visit /docs."}


@app.get("/health", response_model=HealthResponse)
def health():
    sources = {c["source"] for c in state.store}
    return HealthResponse(
        status="ok",
        chunks=len(state.store),
        files=len(sources),
        store_loaded=len(state.store) > 0,
    )


@app.get("/store/detail", response_model=StoreDetail)
def store_detail():
    counts = Counter(c["source"] for c in state.store)
    files = [FileInfo(name=n, chunks=c) for n, c in sorted(counts.items())]
    return StoreDetail(total_chunks=len(state.store), files=files)


# ----- model management -----
@app.get("/models", response_model=ModelsResponse)
def models():
    available = fetch_ollama_models()
    llm_models, embed_models = classify_models(available)

    # always include currently-selected models (cloud models may not be listed)
    if state.llm_model not in llm_models:
        llm_models = [state.llm_model] + llm_models
    if state.embed_model not in embed_models:
        embed_models = [state.embed_model] + embed_models

    return ModelsResponse(
        llm=llm_models,
        embed=embed_models,
        current_llm=state.llm_model,
        current_embed=state.embed_model,
    )


@app.post("/models/llm")
def switch_llm(req: SwitchModelRequest):
    state.llm_model = req.model
    return {"current_llm": state.llm_model}


@app.post("/models/embed")
def switch_embed(req: SwitchModelRequest):
    if state.store:
        raise HTTPException(
            status_code=409,
            detail="Cannot switch embedding model while store has data. Clear the store first.",
        )
    state.embed_model = req.model
    return {"current_embed": state.embed_model}


# ----- stores -----
@app.get("/stores", response_model=list[str])
def list_stores():
    return sorted([p.stem for p in STORE_DIR.glob("*.json")])


@app.post("/stores/{name}/load")
def load_store(name: str):
    path = STORE_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Store '{name}' not found")
    data = json.loads(path.read_text())
    state.store = [{
        "source": r["source"],
        "chunk_index": r["chunk_index"],
        "text": r["text"],
        "embedding": np.array(r["embedding"]),
    } for r in data]
    state.rebuild_bm25()
    state.rebuild_umap()
    return {"loaded": name, "chunks": len(state.store)}


@app.post("/stores/{name}/save")
def save_store(name: str):
    if not state.store:
        raise HTTPException(status_code=400, detail="Store is empty")
    path = STORE_DIR / f"{name}.json"
    data = [{
        "source": c["source"],
        "chunk_index": c["chunk_index"],
        "text": c["text"],
        "embedding": c["embedding"].tolist(),
    } for c in state.store]
    path.write_text(json.dumps(data, indent=2))
    return {"saved": name, "chunks": len(state.store)}


@app.delete("/store")
def clear_store():
    before = len(state.store)
    state.clear()
    return {"cleared": before}


# ----- ingest -----
@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)):
    existing = {c["source"] for c in state.store}
    ingested, skipped = [], []
    new_chunks = 0

    for uf in files:
        if uf.filename in existing:
            skipped.append(uf.filename)
            continue
        data = await uf.read()
        try:
            text = extract_text_from_bytes(uf.filename, data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not text.strip():
            skipped.append(uf.filename)
            continue

        chunks = recursive_chunks(text)
        for i, chunk in enumerate(chunks):
            state.store.append({
                "source": uf.filename,
                "chunk_index": i,
                "text": chunk,
                "embedding": np.array(embed_document(chunk)),
            })
        ingested.append(uf.filename)
        new_chunks += len(chunks)

    state.rebuild_bm25()
    state.rebuild_umap()
    return IngestResponse(
        new_chunks=new_chunks,
        total_chunks=len(state.store),
        files_ingested=ingested,
        files_skipped=skipped,
    )


def _retrieve_and_build(req: AskRequest) -> tuple[list[dict], str]:
    if not state.store or state.bm25 is None:
        raise HTTPException(status_code=400, detail="No documents loaded.")

    q_vec = embed_query(req.query)
    vec = cosine_scores(q_vec, state.store)
    bm = state.bm25.get_scores(req.query)
    hybrid = req.alpha * min_max(vec) + (1 - req.alpha) * min_max(bm)
    top_idx = np.argsort(-hybrid)[:req.top_k]

    hits = [{
        "score": float(hybrid[i]),
        "vec_score": float(vec[i]),
        "bm25_score": float(bm[i]),
        "source": state.store[i]["source"],
        "chunk_index": state.store[i]["chunk_index"],
        "text": state.store[i]["text"],
    } for i in top_idx]

    ctx = "\n\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in hits)
    prompt = PROMPT_TEMPLATE.format(context=ctx, query=req.query)
    return hits, prompt


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    hits, prompt = _retrieve_and_build(req)
    r = ollama.chat(model=state.llm_model, messages=[{"role": "user", "content": prompt}])
    return AskResponse(answer=r["message"]["content"].strip(), sources=hits, prompt=prompt)


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    hits, prompt = _retrieve_and_build(req)

    def token_stream() -> Iterator[bytes]:
        yield (json.dumps({
            "type": "meta",
            "sources": hits,
            "prompt": prompt,
            "llm": state.llm_model,
        }) + "\n").encode("utf-8")

        stream = ollama.chat(
            model=state.llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield (json.dumps({"type": "token", "text": token}) + "\n").encode("utf-8")
        yield (json.dumps({"type": "done"}) + "\n").encode("utf-8")

    return StreamingResponse(token_stream(), media_type="application/x-ndjson")


# ----- UMAP projection -----
class UMAPRequest(BaseModel):
    query: Optional[str] = None


class UMAPPoint(BaseModel):
    source: str
    chunk_index: int
    preview: str
    x: float
    y: float


class UMAPResponse(BaseModel):
    available: bool
    chunks: list[UMAPPoint]
    query_point: Optional[list[float]] = None
    reason: Optional[str] = None


@app.post("/umap", response_model=UMAPResponse)
def umap_project(req: UMAPRequest):
    """Return 2D UMAP coords for every chunk. If a query is provided, also
    returns the query's projected [x, y]. Reducer is fit once per store
    change — see State.rebuild_umap()."""
    if state.umap_reducer is None or state.umap_coords is None:
        return UMAPResponse(
            available=False, chunks=[], query_point=None,
            reason="Need at least 3 chunks to fit UMAP."
        )

    points = [
        UMAPPoint(
            source=c["source"],
            chunk_index=c["chunk_index"],
            preview=c["text"][:140].replace("\n", " "),
            x=float(state.umap_coords[i, 0]),
            y=float(state.umap_coords[i, 1]),
        )
        for i, c in enumerate(state.store)
    ]

    query_point = None
    if req.query:
        try:
            q_vec = embed_query(req.query)
            q_2d = state.umap_reducer.transform(q_vec.reshape(1, -1))
            query_point = [float(q_2d[0, 0]), float(q_2d[0, 1])]
        except Exception:
            query_point = None

    return UMAPResponse(available=True, chunks=points, query_point=query_point)


# ----- mount static frontend -----
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
