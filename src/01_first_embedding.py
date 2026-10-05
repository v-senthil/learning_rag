"""
Milestone 2 — Your first embedding.

Goal: Send a piece of text to Ollama and see the vector that comes back.
Run: python src/01_first_embedding.py
"""

import ollama

# 1. The text we want to embed
text = "Embeddings turn words into numbers that capture meaning."

# 2. Call Ollama's embedding model
response = ollama.embeddings(model="nomic-embed-text", prompt=text)
vector = response["embedding"]

# 3. Inspect what we got back
print(f"Input text: {text}\n")
print(f"Vector type:        {type(vector).__name__}")
print(f"Vector length:      {len(vector)}  (this is the # of dimensions)")
print(f"First 5 numbers:    {vector[:5]}")
print(f"Last 5 numbers:     {vector[-5:]}")
print(f"Smallest value:     {min(vector):.4f}")
print(f"Largest value:      {max(vector):.4f}")
