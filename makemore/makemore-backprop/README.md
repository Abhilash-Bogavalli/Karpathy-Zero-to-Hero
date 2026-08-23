# Makemore: Becoming a Backprop Ninja

Manually deriving and coding every gradient in a BatchNorm MLP — no
`.backward()` — then checking each one against PyTorch's autograd.

---

## What This Is

Following Karpathy's makemore series, this notebook opens up what
`.backward()` actually does: instead of calling it, every gradient in
the network is derived by hand from the forward pass and written out
step by step, then verified against PyTorch's real gradients. The
template (forward pass + exercise structure) is Karpathy's; the
derivations and backward-pass code are my own.

---

## The Four Exercises

### Exercise 1 — Backprop through everything, one variable at a time

Backpropagate through every single intermediate variable in the forward
pass individually — from `dlogprobs` all the way back to `dC` — instead
of using shortcuts. This meant working through the full chain: cross
entropy → logits → the linear layer → BatchNorm → the linear layer →
the embedding lookup.

This was the most approachable exercise once the underlying pattern
clicked: matrix-shaped gradients follow a consistent rule (transpose
and multiply on the correct side), and most of it was direct calculus.
The one genuinely new part was `dC` — accumulating gradients back into
the embedding table by indexing into it and adding, rather than a
single clean matrix operation, since the same character can appear
multiple times across a batch:

```python
dC = torch.zeros_like(C)
for i in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[i,j]
        dC[ix] += demb[i,j]
```

All 26 variables checked exact against PyTorch's autograd.

### Exercise 2 — Cross-entropy backward, in one shot

Instead of backpropagating through each step of cross-entropy
individually (max-subtraction, exp, sum, log, etc.), derive the whole
thing analytically and simplify. The result collapses to a clean,
well-known form:

```python
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n
```

This one was pure calculus with a satisfying result — a lot of
mechanical steps in Exercise 1 collapse into three lines once you
derive the closed form directly instead of stepping through the graph.

### Exercise 3 — BatchNorm backward, in one shot

Same idea as Exercise 2, applied to BatchNorm: derive the gradient of
BatchNorm's output with respect to its input directly, rather than
backpropagating through each intermediate step (mean, variance,
normalization) separately:

```python
dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
```

This was the hardest of the four — needed to look at hints to get the
final derivation right. The BatchNorm backward formula has several
terms that interact (the mean and variance each depend on every
element in the batch, so the gradient has to account for how changing
one input shifts the batch statistics that every other input was
normalized against).

### Exercise 4 — Train the network using only the manual backward pass

Put it all together: train the full MLP end-to-end using the
hand-derived backward pass instead of `.backward()`, confirming the
whole training loop works correctly on manually computed gradients.

This felt redundant relative to the previous three — mechanically
combining exercises 1-3 into a training loop, rather than introducing
new derivation work. Loss dropped from ~3.80 at init to the ~1.9-2.4
range, consistent with the batchnorm run from the previous notebook.

---

## Key Learnings

- Deriving a closed-form gradient (Exercises 2 and 3) versus stepping
  through the computation graph piece by piece (Exercise 1) are two
  different skills — the step-by-step version is more mechanical, the
  closed-form version requires actually working through the calculus
  by hand before writing any code.
- BatchNorm's backward pass is non-trivial precisely because its
  forward pass isn't independent per-example — every example in the
  batch affects the mean/variance every other example is normalized
  against, so the gradient has to account for that coupling.
- Accumulating gradients into an embedding table (`dC`) needs explicit
  indexing and addition, not a single matrix operation, since the same
  row can be touched by multiple examples in a batch.

## References

- [Karpathy's makemore series — Part 4: Becoming a Backprop Ninja](https://youtube.com/...)
- [Batch Normalization paper (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167)

---

## Author

**Abhilash Bogavalli**
[GitHub](https://github.com/Abhilash-Bogavalli)