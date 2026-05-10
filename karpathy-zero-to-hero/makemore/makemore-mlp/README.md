# makemore — MLP

A character-level language model using a Multilayer Perceptron, following Karpathy's makemore series and the Bengio et al. 2003 paper.

## Architecture

- Embedding layer: 27 characters → 10-dimensional embeddings
- Hidden layer: 500 neurons with tanh activation
- Output layer: 27 logits → softmax probabilities
- Loss: Cross-entropy
- Optimiser: SGD with learning rate decay

## Results

| Metric | Value |
|--------|-------|
| Training loss | 2.19 |
| Validation loss | 2.22 |
| Training time | ~9 seconds |

## Key Learnings

- Lookup tables are faster and more expressive than one-hot encoding
- Reducing learning rate over time stabilises loss
- Larger batch/hidden sizes lower training loss but increase train/val gap (overfitting)
- Larger embedding dimensions have the same effect

## Files

- `makemoremlp.ipynb` — full implementation
- `names.txt` — training data

## References

- [Karpathy's makemore MLP lecture](https://www.youtube.com/watch?v=TCH_1BHY58I)
- [Bengio et al. 2003 — A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
