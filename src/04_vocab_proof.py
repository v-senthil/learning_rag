"""
Proof: a token is a whole chunk → one integer.
The individual letters have NO integer ID of their own (unless used alone).

Run: python src/04_vocab_proof.py
"""

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

print("=== Whole words become ONE number each ===\n")
for word in ["pizza", "love", "I", "the"]:
    ids = enc.encode(word)
    print(f"  '{word}'  →  {ids}   (it's just {len(ids)} token)")

print("\n=== Letters alone also get numbers, but DIFFERENT ones ===\n")
for letter in ["p", "i", "z", "a"]:
    ids = enc.encode(letter)
    print(f"  '{letter}'  →  {ids}")

print("\n=== Notice: 'pizza' ID has NOTHING to do with 'p' + 'i' + 'z' + 'z' + 'a' ===")
pizza_id = enc.encode("pizza")
letter_ids = [enc.encode(c)[0] for c in "pizza"]
print(f"  'pizza' alone    : {pizza_id}")
print(f"  'p','i','z','z','a' separately : {letter_ids}")
print(f"\n  → These are independent. The vocab stored 'pizza' as its own entry.")
