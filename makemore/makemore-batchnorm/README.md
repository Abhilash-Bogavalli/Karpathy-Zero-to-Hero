# Makemore Part 3 — Fixing Deep Network Initialization (BatchNorm)

Diagnosing and fixing why a naive deep MLP trains badly at initialization —
from a "confidently wrong" output layer to tanh saturation to BatchNorm.

---

## What This Is

This project follows Karpathy's makemore series, digging into *why* deep
networks train badly with naive initialization, and fixing it step by step —
first with manual initialization tricks, then with Kaiming init, and finally
with Batch Normalization. Also includes a PyTorch-style rewrite of the
network using `Linear`, `BatchNorm1d`, and `Tanh` classes with the same API
shape as `nn.Module`.

---

## The Problems, and What Fixed Each One

### 1. The output layer was confidently wrong at initialization

At init, the output layer (W2, b2) had large random weights, so the initial
logits were large and skewed. Softmax turned that into a very peaked,
wrong probability distribution — the model was confidently guessing the
wrong character from the very first step. This produced a much higher
initial loss than necessary (it should start near the loss of a uniform
guess over 27 characters, ~3.29, not far above it) — visible as the
"hockey stick" loss curve, where the first chunk of training is wasted
just correcting this.

**Fix:** initialize the output layer small — `W2 * 0.01`, `b2 * 0` — so
initial predictions start close to uniform instead of confidently wrong.

### 2. The hidden layer's tanh activations were saturating

Separately from the output layer issue: when inputs to tanh get large
(positive or negative), the output flattens to +1 or -1 — the flat tail
of the tanh curve. The derivative of tanh is `1 - tanh(x)^2`, so near
±1 that derivative collapses toward 0. During backprop, the gradient
gets multiplied by this near-zero derivative, so it becomes ~0 too —
meaning no matter what changes upstream, the network assumes the output
won't change much. Training effectively stalls for those neurons.

**Fix (first attempt):** scale down the weights feeding into tanh —
multiplying by a number less than 1 to keep pre-activations smaller and
away from the saturated region.

### 3. A more principled fix — gain / sqrt(fan_in)

Instead of an arbitrary small multiplier, use a specific formula:
`gain / sqrt(fan_in)`. The `gain` value is not arbitrary either — it's a
known constant tied to the activation function (tanh uses `5/3`,
ReLU uses a different value). Multiplying weights by this factor keeps
the pre-activation distribution closer to Gaussian, which keeps tanh
out of its saturated region without needing to hand-tune a scale factor.

```python
W1 = torch.randn((n_embd * block_size, n_hidden)) * (5/3) / ((n_embd * block_size)**0.5)
```

### 4. The better fix — Batch Normalization

Rather than relying on a fixed initialization scale to keep activations
healthy, normalize them directly. For each neuron, take the mean and std
across the current batch (32 examples), and normalize the pre-activations
to have mean 0 and std 1 before the nonlinearity:

```python
bnmeani = hpreact.mean(0, keepdim=True)
bnstdi = hpreact.std(0, keepdim=True)
hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
```

Since inference often happens on a single example (no batch to compute
statistics from), a running mean and std are tracked throughout training
via a momentum update, and used in place of batch statistics at inference:

```python
bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
bnstd_running  = 0.999 * bnstd_running  + 0.001 * bnstdi
```

### 5. PyTorch-style layer classes

Rebuilt the network using classes (`Linear`, `BatchNorm1d`, `Tanh`) that
mirror the `nn.Module` API — each with `__call__` and `parameters()` —
making the network easier to compose and closer to how real PyTorch
models are structured.

---

## Loss Log

| Fix applied                              | Train loss | Val loss |
|-------------------------------------------|-----------|----------|
| Original                                   | 2.1245    | 2.1682   |
| Fix softmax confidently wrong (output init)| 2.07      | 2.13     |
| Fix tanh saturation at init                 | 2.0356    | 2.1027   |
| Kaiming init (gain / sqrt(fan_in))          | 2.0377    | 2.1070   |
| Add BatchNorm                               | 2.0668    | 2.1048   |

---

## Key Learnings

- The "confidently wrong at init" problem and the "tanh saturation" problem
  are two separate issues with two separate fixes — one is about the output
  layer's initial scale, the other is about the hidden layer's pre-activation
  distribution. Easy to blur together, but they're distinct mechanisms.
- BatchNorm needs a running mean/std because inference often happens on a
  single example, where there's no batch to compute real statistics from.
- Writing my own PyTorch-style `Linear`/`BatchNorm1d`/`Tanh` classes made
  the difference between "I used BatchNorm" and "I understand what it's
  actually doing under the hood."

---

## References

- [Karpathy's makemore series — Part 3: Activations & Gradients,BatchNorm](https://youtu.be/P6sfmUTpUmc?si=nHucVg0qCMMXubet)

---

## Author

**Abhilash Bogavalli**
[GitHub](https://github.com/Abhilash-Bogavalli)