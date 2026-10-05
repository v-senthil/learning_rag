"""
Milestone 3 — Part B: 4 chunking strategies side-by-side.

Runs all 4 strategies on the same doc + same query, so you can SEE
how chunk quality affects retrieval score.

Run: python src/06_chunking_strategies.py
"""

from pathlib import Path
import re
import ollama
import numpy as np


# ---------- helpers ----------
def embed(text: str) -> np.ndarray:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(response["embedding"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ========== Strategy 1: Fixed-Size ==========
def fixed_size_chunks(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Slice every `size` chars with `overlap` between chunks. Dumbest but simplest."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# ========== Strategy 2: Paragraph ==========
def paragraph_chunks(text: str) -> list[str]:
    """Split on double newlines. Respects human structure."""
    return [c.strip() for c in text.split("\n\n") if c.strip()]


# ========== Strategy 3: Recursive Character Splitter ==========
def recursive_chunks(text: str, target: int = 400, overlap: int = 50) -> list[str]:
    """Try separators largest-to-smallest, keeping chunks close to target size.
    Order: paragraphs (\\n\\n) → lines (\\n) → sentences (. ) → words ( )."""
    separators = ["\n\n", "\n", ". ", " "]

    def split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= target:
            return [t]
        if not seps:
            # nothing left to split on → hard cut with overlap
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
                    out.extend(split(part, seps[1:]))   # recurse deeper
                    buffer = ""
                else:
                    buffer = part
        if buffer:
            out.append(buffer)
        return out

    return split(text, separators)


# ========== Strategy 4: Semantic Chunking ==========
def semantic_chunks(text: str, drop_threshold: float = 0.10) -> list[str]:
    """Embed each sentence, insert a boundary where similarity drops sharply.
    A big drop = topic shift = natural chunk break."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) <= 1:
        return sentences

    vecs = [embed(s) for s in sentences]
    sims = [cosine_similarity(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
    avg_sim = sum(sims) / len(sims)

    chunks, current = [], [sentences[0]]
    for i, sim in enumerate(sims):
        if sim < avg_sim - drop_threshold:   # drop below baseline → topic shift
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    chunks.append(" ".join(current))
    return chunks


# ---------- evaluate each strategy on the same query ----------
def evaluate(name: str, chunks: list[str], query_vec: np.ndarray) -> float:
    chunk_vecs = [embed(c) for c in chunks]
    scores = [cosine_similarity(query_vec, cv) for cv in chunk_vecs]
    best_idx = int(np.argmax(scores))
    sizes = [len(c) for c in chunks]

    print(f"\n=== {name} ===")
    print(f"  # chunks:      {len(chunks)}")
    print(f"  chunk sizes:   min={min(sizes)}, avg={sum(sizes)//len(sizes)}, max={max(sizes)}")
    print(f"  top score:     {scores[best_idx]:.4f}")
    print(f"  top chunk:     {chunks[best_idx][:120].replace(chr(10), ' ')}...")
    return scores[best_idx]


def main():
    doc_path = Path(__file__).parent.parent / "data" / "rag_overview.md"
    text = doc_path.read_text()
    query = "Why do we need to split documents into chunks?"

    print(f"Query: '{query}'")
    print(f"Doc length: {len(text)} chars")

    query_vec = embed(query)

    results = {
        "1. Fixed-Size (300/50)":  evaluate("Strategy 1: Fixed-Size (300 chars, 50 overlap)",
                                            fixed_size_chunks(text, 300, 50), query_vec),
        "2. Paragraph":            evaluate("Strategy 2: Paragraph (split on \\n\\n)",
                                            paragraph_chunks(text), query_vec),
        "3. Recursive (target 400)": evaluate("Strategy 3: Recursive (target 400 chars)",
                                              recursive_chunks(text, 400, 50), query_vec),
        "4. Semantic":             evaluate("Strategy 4: Semantic (meaning-shift boundaries)",
                                            semantic_chunks(text, 0.10), query_vec),
    }

    print("\n" + "=" * 60)
    print("FINAL RANKING BY TOP RETRIEVAL SCORE:")
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {score:.4f}  |  {name}")


if __name__ == "__main__":
    main()
