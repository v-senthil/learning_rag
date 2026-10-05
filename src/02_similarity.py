"""
Milestone 2 — Part C: Similarity between sentences.

Goal: Embed multiple sentences and measure how close they are.
Run: python src/02_similarity.py
"""

import ollama
import numpy as np


def embed(text: str) -> np.ndarray:
    """Get an embedding vector from Ollama and return as numpy array."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(response["embedding"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity = angle between two vectors.
      1.0  = same direction (very similar meaning)
      0.0  = unrelated
     -1.0  = opposite meaning

    Formula: (a · b) / (||a|| * ||b||)
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Sentences to test — 3 food, 1 unrelated
sentences = [
    "I love pizza",              # 0
    "I enjoy pasta",             # 1 — similar food topic
    "She loves eating pizza",    # 2 — same topic, different pronoun
    "Quantum physics is hard",   # 3 — completely unrelated
]

# Embed all sentences
print("Embedding sentences...\n")
vectors = [embed(s) for s in sentences]

# Compare sentence 0 against all others
base_idx = 0
base_text = sentences[base_idx]
print(f"Base sentence: '{base_text}'\n")
print(f"{'Similarity':>12}  |  Sentence")
print("-" * 60)

for i, s in enumerate(sentences):
    if i == base_idx:
        continue
    score = cosine_similarity(vectors[base_idx], vectors[i])
    print(f"{score:>12.4f}  |  {s}")
