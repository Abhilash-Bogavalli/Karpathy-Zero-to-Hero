# Text Embeddings and Semantic Search From Scratch

Turn sentences into vectors with a pretrained embedding model, then search by meaning using cosine similarity. This is the retriever half of RAG.

---

## What This Is

A small semantic search engine over 20 sentences I wrote across 5 topics (One Piece, ML, football, dogs, shoes). It is the first building block of a RAG system. Later steps add chunking, a vector store and an LLM.

---

## How It Works

1. **Embed.** `all-MiniLM-L6-v2` turns each sentence into one 384-dimension vector. Inside, it is a transformer with no causal mask (every token sees every token). The token vectors are averaged into one sentence vector.
2. **Compare.** Cosine similarity: dot product divided by the two vector lengths. I wrote it from scratch and checked it against `sentence_transformers`.
3. **Search.** Embed the query, score it against every document, return the top k.
4. **Fast version.** Normalise all document vectors to length 1 once. Then one matrix multiply (`doc_n @ q`) scores everything. Same ranking as the loop version.

The model works because it was trained on a huge number of sentence pairs. Each partner is pulled close and every other sentence is pushed away, using the same cross-entropy loss as makemore.

---

## Results

| Query | Top match | Score | Verdict |
|---|---|---|---|
| Mugiwara has the nicest character | Attacking midfielder is my favourite position | 0.340 | Wrong. Right answer (Luffy) came 2nd at 0.301 |
| Shoes are useless | Sneakers are neat | 0.599 | Right topic, opposite opinion |
| I don't like dogs | Dogs are cool | 0.576 | Right topic, opposite opinion |

- Max input length of the model: [256] tokens. Text past that is silently cut. A long text and its truncated version had cosine similarity [0.5868].

---

## Key Learnings

- Embeddings capture what a sentence is about, not whether it agrees. Negation fails. That is fine for retrieval, because the LLM reads the retrieved text afterwards.
- A low top score means nothing matched well. Real systems set a minimum score and answer "not found" below it.
- Rare words ("Mugiwara") are poorly understood, because the model never learned them.
- Query and documents must use the same model. Each model has its own vector space.
- Input past the max length is silently cut off. This is why documents must be chunked.

---

## How to Run

```bash
pip install sentence-transformers numpy
jupyter notebook embedder.ipynb
```

---

## References

- [sentence-transformers docs](https://www.sbert.net)
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## Author

**Abhilash Bogavalli**
[GitHub](https://github.com/Abhilash-Bogavalli) 