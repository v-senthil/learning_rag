"""
Milestone 3 — Part A: Why we chunk.

Goal: Show empirically that embedding a whole document as ONE vector gives
worse retrieval than embedding it as several focused chunks.

Run: python src/05_why_chunk.py
"""

from pathlib import Path
import ollama
import numpy as np


def embed(text: str) -> np.ndarray:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(response["embedding"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# 1. Load the whole document
doc_path = Path(__file__).parent.parent / "data" / "rag_overview.md"
full_text = doc_path.read_text()

# 2. Split it by blank lines into "paragraph-ish" chunks (naive but illustrative)
chunks = [c.strip() for c in full_text.split("\n\n") if c.strip()]

print(f"Full document length:  {len(full_text)} chars")
print(f"Number of chunks:      {len(chunks)}\n")

# 3. Pick a specific user query
query = "Why do we need to split documents into chunks?"
print(f"Query: '{query}'\n")

# 4. Compare: one-vector-for-whole-doc  vs  one-vector-per-chunk
query_vec = embed(query)

# Score 1: embed entire document as ONE vector
whole_doc_vec = embed(full_text)
whole_doc_score = cosine_similarity(query_vec, whole_doc_vec)

# Score 2: embed each chunk, take the BEST match
chunk_vecs = [embed(c) for c in chunks]
chunk_scores = [cosine_similarity(query_vec, cv) for cv in chunk_vecs]
best_idx = int(np.argmax(chunk_scores))
best_chunk_score = chunk_scores[best_idx]

# 5. Report
print(f"{'Strategy':<30} | Score")
print("-" * 60)
print(f"{'Embed WHOLE doc as 1 vector':<30} | {whole_doc_score:.4f}")
print(f"{'Best matching chunk':<30} | {best_chunk_score:.4f}")
print()
print("All chunk scores (ranked):")
ranked = sorted(enumerate(chunk_scores), key=lambda x: -x[1])
for idx, score in ranked:
    preview = chunks[idx][:70].replace("\n", " ")
    marker = " ← BEST" if idx == best_idx else ""
    print(f"  {score:.4f}  | {preview}...{marker}")
