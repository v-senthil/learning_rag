"""
Milestone 4 — Debug fix A: rebuild the vector store using nomic-embed-text's
native 'search_document:' prefix for indexed chunks.

nomic-embed-text is trained with asymmetric prefixes to solve the
question-vs-statement mismatch:
   Queries  → 'search_query: <text>'
   Chunks   → 'search_document: <text>'

This script re-embeds every chunk with the document prefix and saves to
vector_store_v2.json so we can compare old vs new.

Run: python src/09_rebuild_with_prefixes.py
"""

import json
from pathlib import Path
import ollama


DATA_DIR = Path(__file__).parent.parent / "data"
STORE_PATH = Path(__file__).parent.parent / "vector_store_v2.json"
CHUNK_TARGET = 400
CHUNK_OVERLAP = 50
DOC_PREFIX = "search_document: "


def recursive_chunks(text: str, target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]

    def split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= target:
            return [t]
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
        return out

    return split(text, separators)


def embed_document(text: str) -> list[float]:
    """Prepend 'search_document: ' so nomic knows this is an indexed passage."""
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=DOC_PREFIX + text,
    )
    return response["embedding"]


def main():
    md_files = sorted(DATA_DIR.glob("*.md"))
    print(f"Found {len(md_files)} markdown files\n")
    print(f"Using prefix: '{DOC_PREFIX}'\n")

    store = []
    for path in md_files:
        text = path.read_text()
        chunks = recursive_chunks(text)
        print(f"  {path.name:<35} → {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            store.append({
                "source": path.name,
                "chunk_index": i,
                "text": chunk,                    # store raw text (no prefix)
                "embedding": embed_document(chunk),  # but embed WITH prefix
            })

    STORE_PATH.write_text(json.dumps(store, indent=2))
    print(f"\n✅ Saved {len(store)} chunks → {STORE_PATH.name}")


if __name__ == "__main__":
    main()
