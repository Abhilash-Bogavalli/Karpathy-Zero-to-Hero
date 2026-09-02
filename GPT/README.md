# Building GPT From Scratch

A character-level GPT — from a raw bigram model up through multi-head
self-attention, feedforward blocks, residual connections, and dropout —
built and trained from scratch in PyTorch.

---

## What This Is

Following Karpathy's "Let's build GPT" video, this repo implements a
decoder-only transformer step by step, starting from the simplest
possible baseline (a bigram model) and adding one real architectural
idea at a time, so each addition can be understood on its own before
the next one stacks on top.

---

## The Build, In Order

### 1. Bigram baseline

The starting point: predict the next character using only an embedding
table and a linear layer, with no context beyond the current token.
This establishes the training loop, data pipeline, and generation logic
that everything after builds on.

### 2. Position embeddings

Added a second embedding table indexed by *position* (0 through
`block_size-1`), summed with the token embedding. Without this, the
model has no notion of word order — attention itself is
permutation-invariant, so position has to be injected into the input
explicitly.

### 3. Single attention head — the first step toward self-attention

Built a `Head` module with three separate learned projections of the
input: **key**, **query**, and **value**. The query and key are matrix
multiplied to produce a `(T, T)` matrix — `wei[i, j]` represents *how
much position i attends to position j*, i.e. how well position i's
query matches position j's key.

```python
wei = q @ k.transpose(-2, -1) * C ** -0.5
```

The `* C ** -0.5` term scales the dot product by `1/sqrt(head_size)`.
Without it, larger head sizes produce higher-variance dot products,
which pushes softmax into an overly peaked, saturated regime — the
same underlying problem as tanh saturation, just showing up in
attention scores instead of activations. Scaling keeps the softmax
output "diffuse" so gradients keep flowing to more than one position.

Since a position must never attend to positions that come *after* it
(it can't be trained to predict the future), a lower-triangular mask
sets all disallowed entries to `-inf` before the softmax, so they
contribute exactly zero attention weight:

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
```

The mask itself (`tril`) is stored via `register_buffer` rather than
as a normal attribute — it needs to move with the model when calling
`.to(device)`, but it's a fixed structural mask, not a learnable
weight, so the optimizer should never update it.

Finally, the attention weights are used to combine the **values**
(not the raw input, and not the keys) of every attended-to position:

```python
out = wei @ v
```

### 4. Multi-head attention

Instead of one attention head, run several independently-learned heads
in parallel, each operating on a smaller slice (`head_size = n_embd //
n_head`), then concatenate their outputs back to the full embedding
dimension. Each head can learn to attend to a different kind of
relationship between positions, rather than all attention being forced
through a single mechanism.

```python
self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
self.proj = nn.Linear(n_embd, n_embd)
```

Concatenation alone just places the heads' outputs side by side without
letting them interact — the `proj` layer afterward is what actually
mixes information across heads into a single combined representation.

### 5. FeedForward

After attention gathers information from other positions, a small
per-position MLP (expand to `4*n_embd`, ReLU, project back down) lets
the model further process what attention collected, rather than passing
the raw attention output straight through.

### 6. Block — attention + feedforward, with LayerNorm and residual connections

Each `Block` combines multi-head attention and a feedforward layer,
each preceded by `LayerNorm` (to normalize inputs before each
sub-layer) and wrapped in a residual (skip) connection:

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

The `x + ...` pattern gives gradients a direct path backward through
each sub-layer during backprop, rather than forcing them through every
transformation in sequence — the same mechanism that motivates ResNet's
skip connections, applied here to prevent vanishing gradients as the
network gets deeper (6 blocks stacked).

### 7. Dropout

20% of activations are randomly zeroed during training (in attention
weights, the projection output, and the feedforward output), forcing
the model to not over-rely on any single pathway — improving
generalization and reducing overfitting.

---

## Results

| Setup | Validation Loss |
|---|---|
| This build (hardware-limited: `n_embd=36`, `n_head=6`, `n_layer=6`) | 1.9 |
| Karpathy's reference run (larger params) | 1.4 |

The gap is attributable to reduced model size for local hardware
constraints, not a difference in implementation — the architecture
itself (attention, masking, scaling, multi-head, residual blocks) is
built and verified from scratch as above.

---

## Key Learnings

- Attention direction matters precisely: `wei[i,j]` is "how much
  position i attends to position j" — the query position looking
  outward at key positions, not the reverse.
- The `1/sqrt(head_size)` scaling in attention solves the same problem
  as Kaiming initialization does for weights — controlling variance to
  keep softmax (or, in that case, tanh) from saturating.
- Residual connections exist for the same reason in transformers as in
  ResNet: giving gradients a direct backward path so they don't have to
  flow through every stacked layer to reach earlier weights.
- Concatenating multi-head outputs isn't enough on its own — a
  projection layer afterward is what actually lets the heads' learned
  information mix together.

---

## How to Run

```bash
git clone https://github.com/Abhilash-Bogavalli/Karpathy-Zero-to-Hero.git
cd GPT
python gpt.py
```

---

## References

- [Karpathy — Let's build GPT: from scratch, in code, spelled out](https://youtube.com/...)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)

---

## Author

**Abhilash Bogavalli**
[GitHub](https://github.com/Abhilash-Bogavalli) 