"""
Milestone 2 — Bonus: See how text is broken into tokens.

We use a common BPE tokenizer (tiktoken, used by GPT models) for illustration.
The exact tokens differ per model, but the IDEA is universal.

Install once:  pip install tiktoken
Run:           python src/03_tokenization_peek.py
"""

import tiktoken

# cl100k_base is the tokenizer used by GPT-4 / GPT-3.5
# nomic-embed-text uses a different one, but the behavior is the same idea.
enc = tiktoken.get_encoding("cl100k_base")

sentences = [
    "I love pizza",
    "piezoelectric",           # rare word — split into pieces
    "unbelievable",
    "RAG uses embeddings",
    "asdfghjkl",                # nonsense — forced to split
]

for s in sentences:
    token_ids = enc.encode(s)
    token_strs = [enc.decode([tid]) for tid in token_ids]
    print(f"Input:  '{s}'")
    print(f"  Token IDs: {token_ids}")
    print(f"  Pieces:    {token_strs}")
    print(f"  Count:     {len(token_ids)} tokens\n")
