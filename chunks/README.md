# Chunking Strategies for RAG: Compared on a Real Article

I cut a long Wikipedia article into chunks three different ways and measured which way lets a search find the right answer.

---

## Why Chunk?

The embedding model (`all-MiniLM-L6-v2`) reads about 256 tokens at most. A whole article as one vector is a blur of many topics. So the article is cut into chunks of 100-200 tokens, and each chunk gets its own vector.

---

## The Three Chunkers

1. **Fixed-size:** cut every 120 words. Each chunk repeats the last 30 words of the previous one (overlap), so an answer on a boundary still sits whole in one chunk.
2. **Sentence-packing:** add whole sentences to a chunk until the 150-token limit is reached, then start a new chunk. It repeats the last sentence as overlap.
3. **Recursive:** pack whole paragraphs into a chunk up to the limit. If a single piece is too big, split it by the next separator: lines, then sentences, then words.

---

## How I Tested

- Document: the Wikipedia article "Artificial intelligence" (text not included here)
- 16 questions, written in my own words. Each has an answer phrase that must appear in the right chunk.
- For each chunker: embed all chunks, embed the question, take the top 3 chunks by similarity. It is a hit if one of them contains the answer phrase.
- Later I allowed more than one accepted phrase per question, because the article sometimes states a fact in two places.

---

## Results

| Chunker | Chunks | Largest chunk (tokens) | Hit@3 at first | Hit@3 after fixing the test |
|---|---|---|---|---|
| Fixed | 147 | 190 | 0.9375 (15/16) | 1.0 (16/16) |
| Sentence-packing | 171 | 255 | 0.8125 (13/16) | 0.875 (14/16) |
| Recursive | 201 | 150 | 0.875 (14/16) | 0.9375 (15/16) |

---

## What I Found

- **My test was too narrow.** The 2012 GPU question counted as a miss, but the top-ranked chunk answered it correctly. It just didn't contain my one phrase. Allowing several phrases raised every score by one question. When a score looks bad, read the misses before blaming the system.
- **One paragraph kept causing misses.** The article's opening summary packs many years and topics into a few sentences, so its chunk is a blur. The 1956, 2012 and 2017 questions kept missing there. This is the "chunk too big" problem.
- **Most misses were near misses**, at rank 4 to 7. Retrieving more than 3 chunks would catch them.
- **No answer was cut in half by a boundary,** but my answer phrases are very short, so this test says little about boundary damage.
- **The largest sentence-packing chunk was 255 tokens,** right at the model's 256 limit, so its end may be cut off.
- **The strategies are 1-2 questions apart out of 16.** That is too small to call a winner. This is one document and 16 questions.

---

## Author

**Abhilash Bogavalli**
[GitHub](https://github.com/Abhilash-Bogavalli) 