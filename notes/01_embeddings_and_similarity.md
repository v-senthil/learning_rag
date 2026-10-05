# 01 — Embeddings and Similarity

**Date:** 2026-04-20
**Stage:** 2 (First LLM app — embeddings → RAG)
**Milestone:** #2 of 7 (Embeddings 101)
**Scripts:** `src/01_first_embedding.py`, `src/02_similarity.py`, `src/03_tokenization_peek.py`, `src/04_vocab_proof.py`

---

## 1. What I Built / Learned

I embedded my first piece of text using Ollama's `nomic-embed-text` model and saw
the 768-number vector that represents it. I then embedded 4 sentences and used
cosine similarity to measure how close they sit in "meaning space" — proving with
numbers that embeddings capture *meaning*, not surface-level words. Along the way
I broke open the black box of how a word becomes numbers: tokenization → vocab
lookup → embedding matrix → transformer layers → pooling. I cleared up a big
misconception (vocab maps whole chunks, NOT letters) and understood why
"pizza vs pasta" scores higher than "pizza vs quantum physics" — even with no
shared words.

---

## 2. The Child-Level Explanation (the permanent reference)

### 🧒 Child's first question
> "A computer is just electricity and numbers. How does it even know what
> 'pizza' is?"

**Honest answer:** It doesn't. It never does. It only ever sees numbers.
The whole trick is how we turn text into the *right* numbers so that similar
meanings produce similar number patterns.

### The full pipeline — every step

```
  "I love pizza"
        ↓
  [1] Tokenization        → ["I", "love", "pizza"]
        ↓
  [2] Vocab Lookup        → [40, 1842, 13142]         (token → integer ID)
        ↓
  [3] Embedding Matrix    → 3 rows of 768 numbers     (one vector per token)
        ↓
  [4] Transformer Layers  → context-aware vectors     (tokens look at each other)
        ↓
  [5] Pooling             → 1 vector of 768 numbers   (sentence-level vector)
        ↓
  [6] Normalization       → the final embedding
```

---

### 🔹 Step 1 — Tokenization

**Child:** "You split the sentence into words, right?"
**Real answer:** Close, but not always into words — often into **sub-word pieces**.

`"unbelievable"` doesn't become `["unbelievable"]`. It becomes
`["un", "believ", "able"]`. Why?
- The vocabulary is a **fixed size** (typically 30k–50k pieces).
- Breaking into reusable chunks means the model can understand words it has
  **never seen before** by combining known sub-pieces.

The algorithm used is usually **BPE (Byte-Pair Encoding)** or **WordPiece**.
Both work by statistically finding common byte sequences during training and
locking them in as "tokens."

> **Aha:** "pizza" is likely 1 token (common word). "piezoelectric" might be
> 3–4 tokens. More tokens = the model sees the word as pieces, not a whole.

---

### 🔹 Step 2 — Vocab Lookup (Token → Integer)

**Child:** "Okay, so now we have pieces. How do those become numbers?"

The model has a **dictionary** — a big lookup table mapping every token to a
fixed integer ID:

```
"the"    → 278
"pizza"  → 13142
"love"   → 1842
"I"      → 40
```

So `"I love pizza"` becomes `[40, 1842, 13142]`. That's it. The model doesn't
"know" these are words anymore — just integers.

> **Common misconception:** "Token 40 and 41 must be related, right?"
> **No.** The integer ID is arbitrary — just an address, like a house number.
> House 40 and house 41 aren't related by being neighbors.

---

### 🔹 My Big Misconception (and the correction)

**What I thought initially:**
> "If p = 1, i = 3, z = ?, a = ? — how is z = 1 if p is already 1? How can the
> integer be fixed?"

**The correction:**
The vocab is **NOT a letter-to-number map**. It's a **whole-chunk to number map**.

Think of it like a **phonebook** — each *complete entry* gets one row:

```
Row 40     : "I"
Row 1842   : "love"
Row 13142  : "pizza"      ← the ENTIRE word is ONE entry, not p+i+z+z+a
Row 40000  : "piez"       ← only used when needed for rare words
```

**"pizza" as a whole = 13142.** It has nothing to do with the letters
p, i, z, z, a individually. The model never sees the letters. It sees the
whole chunk at once.

**Library analogy:** Imagine a library with 50,000 books, each on its own
numbered shelf. When you say "I love pizza", the librarian fetches 3 books:
shelf 40, shelf 1842, shelf 13142. She doesn't look at the letters on the
cover — she just cares about which shelf.

---

### 🔹 Step 3 — The Embedding Matrix (THE critical step)

**Child:** "So where do the 768 numbers come from?"

The model has a **giant table** — the embedding matrix — with one row per
token in the vocab:

```
              col_0    col_1    col_2   ...   col_767
row 0     :  -0.04    0.12    -0.88   ...    0.33
row 1842  :   0.11   -0.33     0.45   ...    0.92   ← "love"
row 13142 :   0.45    0.67    -0.22   ...    0.15   ← "pizza"
...
row 50000 :   ...
```

Token ID 13142 (pizza) means **"go to row 13142 and grab those 768 numbers."**
That's it. A lookup. No math. No AI yet.

> **Aha:** These 768 numbers per row were **learned during training**. When
> the model was trained on billions of sentences, it slowly nudged these
> numbers so that tokens used in similar contexts ended up with similar rows.
> "pizza" and "pasta" naturally ended up pointing in similar directions —
> because they appeared near similar words like "eat", "delicious",
> "restaurant".

**The meaning is baked into the matrix.**

**Key distinction:**
- `13142` = **address** (meaningless — just a lookup key)
- row 13142 of the matrix = **meaning** (the 768 learned numbers)

---

### 🔹 Step 4 — Transformer Layers (Context Matters)

**Child:** "But 'bank' means different things in 'river bank' vs 'money bank' —
how does the number know?"

Great instinct. Step 3 gives you the **same vector for "bank"** regardless of
context. That's not enough.

So the vectors go through **transformer layers** (typically 6–24 stacked).
Each layer lets every token **look at every other token** and update its own
vector based on what's around it. This is **self-attention**.

- "bank" asks: "Who around me should I pay attention to?"
- If "river" is nearby, "bank" shifts toward the "geography" region.
- If "money" is nearby, "bank" shifts toward the "finance" region.

After 12 layers of this back-and-forth, every token's vector is **deeply
aware of its neighbors**.

> **Aha:** The word "pizza" in "I love pizza" vs "pizza box is empty" gets
> **different final vectors** — same starting point, different context,
> different endings.

---

### 🔹 Step 5 — Pooling (Many Vectors → One)

**Child:** "We now have 3 vectors for 'I love pizza'. You said embeddings are
ONE vector. What happened?"

We combine the per-token vectors into a single sentence-level vector:

1. **Mean pooling** (most common, used by `nomic-embed-text`) — average all
   token vectors.
2. **CLS token** — some models add a special `[CLS]` token and use its final
   vector.
3. **Max pooling** — take the max value per dimension.

For `nomic-embed-text`: `vector = mean(token_vectors)` → one 768-dim vector.

---

### 🔹 Step 6 — Normalization (Optional Cleanup)

Some models divide the vector by its length so all vectors have magnitude 1.0,
which makes cosine similarity faster. `nomic-embed-text` returns **unnormalized**
vectors (my output had values from -3.34 to +2.34 — proof they weren't
normalized).

---

### 🧠 One-Line Summary of the Whole Pipeline

> A word becomes an embedding by: **split into tokens → look up each token's
> row in a big learned table → let tokens update each other based on context →
> average them into one vector.**

---

## 3. Cosine Similarity — The Ruler of Meaning

### Intuition
Each embedding is an **arrow pointing in 768-dimensional space**. Cosine
similarity measures the **angle** between two arrows:

- Arrows pointing the **same way** → similarity ≈ 1.0 (same meaning)
- Arrows at **90°** → similarity ≈ 0.0 (unrelated)
- Arrows pointing **opposite ways** → similarity ≈ -1.0 (opposite meaning)

**Why angle, not distance?** Magnitude doesn't matter — only direction encodes
meaning. A long sentence and a short sentence about the same topic point the
same way.

### Formula

```
cosine_sim(a, b) = (a · b) / (||a|| × ||b||)
```

| Part | What it does |
|---|---|
| `np.dot(a, b)` | Dot product — multiplies corresponding numbers and sums them. Big positive = vectors point same direction. |
| `np.linalg.norm(a)` | Length of vector `a` (Pythagorean theorem, but in 768D). |
| Dividing by the norms | Cancels out magnitude — we only care about direction. |

Result: a number between -1 and +1 that tells us how aligned two vectors are.

---

## 4. The Code

### `src/01_first_embedding.py` — first embedding
Sends text to Ollama, inspects what comes back.

```python
import ollama

text = "Embeddings turn words into numbers that capture meaning."
response = ollama.embeddings(model="nomic-embed-text", prompt=text)
vector = response["embedding"]

print(f"Vector length:   {len(vector)}")   # 768
print(f"First 5 numbers: {vector[:5]}")
print(f"Smallest value:  {min(vector):.4f}")
print(f"Largest value:   {max(vector):.4f}")
```

### `src/02_similarity.py` — measuring similarity

```python
import ollama
import numpy as np

def embed(text: str) -> np.ndarray:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(response["embedding"])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

sentences = [
    "I love pizza",              # 0 — base
    "I enjoy pasta",             # 1 — similar food topic
    "She loves eating pizza",    # 2 — same topic, different pronoun
    "Quantum physics is hard",   # 3 — completely unrelated
]
vectors = [embed(s) for s in sentences]

for i, s in enumerate(sentences[1:], start=1):
    score = cosine_similarity(vectors[0], vectors[i])
    print(f"{score:.4f} | {s}")
```

### `src/04_vocab_proof.py` — proving whole-word tokenization

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# Whole words → ONE integer each
for word in ["pizza", "love", "I", "the"]:
    print(f"'{word}' → {enc.encode(word)}")

# Letters alone → different, unrelated integers
for letter in ["p", "i", "z", "a"]:
    print(f"'{letter}' → {enc.encode(letter)}")
```

---

## 5. What I Observed When I Ran It

### First embedding (`01_first_embedding.py`)
```
Input text: Embeddings turn words into numbers that capture meaning.

Vector type:        list
Vector length:      768  (this is the # of dimensions)
First 5 numbers:    [0.2974, 0.2162, -3.3475, -1.1179, 1.7223]
Last 5 numbers:     [0.2329, -1.4546, -1.4537, -0.7038, -0.5127]
Smallest value:     -3.3475
Largest value:      2.3392
```

My reaction: *"Okay, so it's literally just a list of 768 floats. The numbers
look random. Clearly the meaning isn't in any one number — it must be in the
pattern."*

### Similarity test (`02_similarity.py`)
```
Base sentence: 'I love pizza'

  Similarity  |  Sentence
------------------------------------------------------------
      0.8719  |  She loves eating pizza       ← HIGHEST
      0.6240  |  I enjoy pasta
      0.4321  |  Quantum physics is hard      ← LOWEST
```

My reaction: *"Wait — 'She loves eating pizza' scored HIGHER than 'I enjoy
pasta'? Even though 'I enjoy pasta' shares the word 'I' with the base?
That's exactly the opposite of what I predicted."*

---

## 6. What Surprised Me (the "Aha" moments)

1. **Pronouns barely matter.** I thought sharing the word "I" would boost
   similarity. It didn't — changing "I" → "She" barely moved the score
   because content words (pizza, love) dominate.

2. **"Quantum physics is hard" scored 0.43, not 0.** I expected "unrelated" to
   be near zero. Two reasons it isn't:
   - `nomic-embed-text` doesn't center vectors at zero — all sentences tend to
     skew in a roughly similar direction.
   - Both sentences share structural features (short English statements,
     subject-verb-object form).
   **Takeaway:** compare cosine scores *relative* to each other, not in
   absolute terms. 0.87 > 0.62 > 0.43 is what matters.

3. **"Pizza" is one token, not five letters.** I was sure the model assembled
   meanings letter-by-letter. In reality, the vocab stores whole chunks as
   single entries. Letters only matter when the word is rare and must be
   broken down.

---

## 7. What I Got Wrong at First

| Misconception | Correction |
|---|---|
| "Letters each get an integer; words are just the sum of letter-integers." | The vocab maps **whole chunks** to integers. 'pizza' = one number. Letters have their own unrelated numbers. |
| "Sharing common words like 'I' should make sentences similar." | Function words (I, the, a) barely affect embeddings. Content words dominate. |
| "The integer ID itself must carry meaning — nearby IDs must be similar." | The integer is just an **address** (arbitrary). Meaning lives in the **row of 768 numbers** that address points to. |
| "Unrelated things will have cosine similarity around 0." | Not for this model — baseline "unrelated" is ~0.4. Read scores relatively, not absolutely. |

---

## 8. Follow-up Questions I Still Have

1. **How is the embedding matrix actually *trained*?** What algorithm nudges
   "pizza" and "pasta" rows toward each other? (Contrastive loss? Masked
   language modeling? Both?)
2. **Why 768 dimensions specifically?** Why not 100 or 10,000?
3. **If transformer layers modify per-token vectors, does that mean each
   sentence goes through a full forward pass every time I embed it?** How
   expensive is that?
4. **Does `nomic-embed-text` have a separate "query" vs "document" mode?**
   I've heard some embedding models produce different vectors depending on
   whether you're searching or indexing.
5. **Why do "unrelated" sentences cluster at ~0.4 instead of 0?** Is this a
   quirk of this model, or common to all? Would normalizing help?

---

## 9. Where This Fits in the Roadmap

- **Stage:** 2 (First LLM app — embeddings → RAG)
- **Completed milestones:**
  - ✅ Setup + sample markdown knowledge base (Milestone 1)
  - ✅ First embedding + similarity math (Milestone 2 — this note)
- **Up next:** Milestone 3 — Chunking strategies. Before we embed our 5
  markdown files, we need to decide how to split them (whole doc vs
  fixed-size chunks vs semantic chunks). Chunking is the single
  highest-leverage decision in a RAG system.

**Why this matters for the RAG system:**
Every piece of what we just learned is the foundation of retrieval. When a
user asks a question, we'll:
1. Embed the question using this same pipeline
2. Compute cosine similarity against every chunk in our vector store
3. Return the top-k most similar chunks as context for the LLM

Without understanding *why* "pizza" and "pasta" cluster together, I'd have no
way to debug bad retrievals later.

---

## 10. Commands I Ran

```bash
# Setup
cd ~/Documents/AILearning/rag_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install tiktoken

# Scripts
python src/01_first_embedding.py
python src/02_similarity.py
python src/04_vocab_proof.py
```
