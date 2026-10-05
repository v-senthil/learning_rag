# Transformer Architecture

## The Breakthrough

Introduced in the 2017 paper "Attention Is All You Need", the Transformer
replaced recurrent networks (RNNs, LSTMs) as the dominant architecture for
sequence modeling. It powers nearly every modern LLM — GPT, Llama, Claude,
Gemini, and more.

## Core Building Blocks

### Self-Attention
Every token in the input "looks at" every other token and decides which ones
to pay attention to. This is done via three learned projections: Query (Q),
Key (K), and Value (V). The attention score is:

    softmax(Q · K^T / sqrt(d_k)) · V

This lets the model capture long-range dependencies — something RNNs struggled
with.

### Multi-Head Attention
Instead of one attention mechanism, we run several in parallel ("heads"), each
learning a different relationship. Their outputs are concatenated and
projected.

### Feed-Forward Layers
After attention, each token goes through a small MLP independently. This is
where most of the model's parameters live.

### Positional Encoding
Transformers have no inherent notion of order (unlike RNNs). We inject
position info via sinusoidal encodings or learned embeddings.

## Encoder vs Decoder

- **Encoder-only** (BERT): great for understanding tasks — classification,
  embeddings.
- **Decoder-only** (GPT, Llama): great for generation — chat, completion.
- **Encoder-decoder** (T5, original Transformer): translation, summarization.

## Why It Scales So Well

Transformers are highly parallelizable (unlike RNNs) which means they can be
trained on massive datasets with GPUs efficiently. This is the reason we see
models with billions of parameters today.
