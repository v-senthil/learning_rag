"""
Milestone 7 — Reusable ingestion tool.

Takes any mix of .pdf / .md / .txt files (or a directory), chunks them,
embeds with nomic-embed-text's 'search_document:' prefix, and saves a
JSON vector store that 18_ask.py can query.

Usage:
  # single file
  python src/17_ingest.py path/to/paper.pdf

  # multiple files, explicit output
  python src/17_ingest.py notes.md paper.pdf --output my_store.json

  # entire directory (recursive)
  python src/17_ingest.py path/to/docs --recursive

  # append to existing store instead of overwriting
  python src/17_ingest.py new_file.pdf --output my_store.json --append
"""

import argparse
import json
from pathlib import Path
import ollama
from pypdf import PdfReader


EMBED_MODEL = "nomic-embed-text"
DOC_PREFIX = "search_document: "
DEFAULT_OUTPUT = Path(__file__).parent.parent / "my_store.json"
SUPPORTED_EXTS = {".pdf", ".md", ".txt"}


# ---------- recursive character splitter (same as earlier milestones) ----------
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


# ---------- extract text based on file type ----------
def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n\n".join(pages)
    elif ext in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file extension: {ext} ({path.name})")


def embed_document(text: str) -> list[float]:
    r = ollama.embeddings(model=EMBED_MODEL, prompt=DOC_PREFIX + text)
    return r["embedding"]


# ---------- collect files from args (files and/or dirs) ----------
def gather_files(paths: list[str], recursive: bool) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  ⚠️  skip {path} (not found)")
            continue
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTS:
                out.append(path)
            else:
                print(f"  ⚠️  skip {path.name} (unsupported extension)")
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in sorted(path.glob(pattern)):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                    out.append(f)
    return out


def main():
    p = argparse.ArgumentParser(description="Ingest files into a hybrid RAG store.")
    p.add_argument("paths", nargs="+", help="file(s) or directory to ingest")
    p.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT,
                   help=f"output JSON store path (default: {DEFAULT_OUTPUT.name})")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="recurse into directories")
    p.add_argument("--append", "-a", action="store_true",
                   help="append to existing store instead of overwriting")
    p.add_argument("--chunk-target", type=int, default=400,
                   help="target chunk size in characters (default: 400)")
    p.add_argument("--chunk-overlap", type=int, default=50,
                   help="overlap between chunks in characters (default: 50)")
    args = p.parse_args()

    files = gather_files(args.paths, args.recursive)
    if not files:
        print("❌ No supported files found.")
        return

    # load existing store if appending
    if args.append and args.output.exists():
        store = json.loads(args.output.read_text())
        print(f"📦 Loaded {len(store)} existing chunks from {args.output.name}")
    else:
        store = []

    existing_sources = {c.get("source") for c in store}

    print(f"\nFound {len(files)} file(s) to ingest:\n")
    total_new = 0
    for path in files:
        if path.name in existing_sources:
            print(f"  ⏭️  {path.name} — already in store, skipping")
            continue

        try:
            text = extract_text(path)
        except Exception as e:
            print(f"  ❌ {path.name} — extraction failed: {e}")
            continue

        if not text.strip():
            print(f"  ⚠️  {path.name} — no extractable text (empty or image-only PDF?)")
            continue

        chunks = recursive_chunks(text, args.chunk_target, args.chunk_overlap)
        print(f"  ✅ {path.name:<40} {len(text):>6} chars → {len(chunks)} chunks  (embedding...)")

        for i, chunk in enumerate(chunks):
            store.append({
                "source": path.name,
                "chunk_index": i,
                "text": chunk,
                "embedding": embed_document(chunk),
            })
        total_new += len(chunks)

    args.output.write_text(json.dumps(store, indent=2))
    print(f"\n✨ Saved {len(store)} total chunks ({total_new} newly added) → {args.output}")
    print(f"   Ask questions with:  python src/18_ask.py --store {args.output.name}")


if __name__ == "__main__":
    main()
