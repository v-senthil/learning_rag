"""
Milestone 3 — Part C: chunk ALL markdown files and build a mini vector store.

For each .md file in data/:
  1. Read the file
  2. Recursively chunk it (target ~400 chars, 50 char overlap)
  3. Embed each chunk with Ollama
  4. Store chunk metadata + vector in memory
Finally, write everything to vector_store.json — this is our tiny "database"
that Milestone 4 will query.

Run: python src/07_chunk_all_docs.py
"""

import json
from pathlib import Path
import ollama


DATA_DIR = Path(__file__).parent.parent / "data"
STORE_PATH = Path(__file__).parent.parent / "vector_store.json"
CHUNK_TARGET = 400     # aim for chunks near this size (chars)
CHUNK_OVERLAP = 50     # overlap between consecutive chunks (chars)


# ---------- recursive character splitter (same algorithm as Milestone 3 Part B) ----------
def recursive_chunks(text: str, target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Try separators largest → smallest, stay near target size.
    Order: paragraphs (\\n\\n) → lines (\\n) → sentences (. ) → words ( )."""
    separators = ["\n\n", "\n", ". ", " "]

    def split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= target:
            return [t]
        if not seps:
            # no more separators — hard cut with overlap as last resort
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
                    out.extend(split(part, seps[1:]))   # recurse with next separator
                    buffer = ""
                else:
                    buffer = part
        if buffer:
            out.append(buffer)
        return out

    return split(text, separators)


def embed(text: str) -> list[float]:
    """Get embedding as a plain list (JSON-serializable)."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def main():
    md_files = sorted(DATA_DIR.glob("*.md"))
    print(f"Found {len(md_files)} markdown files in {DATA_DIR.name}/\n")

    store = []                    # list of chunk records
    per_file_counts = {}

    for path in md_files:
        text = path.read_text()
        chunks = recursive_chunks(text)
        per_file_counts[path.name] = len(chunks)
        print(f"  {path.name:<35} {len(text):>5} chars → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            store.append({
                "source": path.name,
                "chunk_index": i,
                "text": chunk,
                "embedding": embed(chunk),
            })

    STORE_PATH.write_text(json.dumps(store, indent=2))

    # ---------- summary ----------
    total = len(store)
    avg_size = sum(len(c["text"]) for c in store) // total
    min_size = min(len(c["text"]) for c in store)
    max_size = max(len(c["text"]) for c in store)

    print("\n" + "=" * 60)
    print(f"✅ Saved {total} chunks → {STORE_PATH.name}")
    print(f"\nPer-file breakdown:")
    for name, count in per_file_counts.items():
        print(f"  {name:<35} {count} chunks")
    print(f"\nChunk size: min={min_size}, avg={avg_size}, max={max_size} chars")
    print(f"Store file size: {STORE_PATH.stat().st_size / 1024:.1f} KB")
    print(f"Embedding dim: {len(store[0]['embedding'])}")


if __name__ == "__main__":
    main()
