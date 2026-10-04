# BPE Tokenizer From Scratch

A byte-level Byte-Pair Encoding tokenizer (train, encode, decode) built from scratch in Python and compared against OpenAI's `tiktoken`.

---

## What This Is

Following Karpathy's "Let's build the GPT Tokenizer", I built the step that happens before text ever reaches an LLM. Models don't read characters or words. They read integer token IDs. This repo builds the algorithm that decides what those tokens are.

---

## How It Works

### 1. Text becomes bytes

Unicode gives every character a number called a code point: `ord("A")` is 65, `ord("😀")` is 128512. There are about 150,000 code points, which is too many for a vocabulary.

UTF-8 turns each code point into 1 to 4 bytes, each between 0 and 255. So any text in any language becomes a list of numbers from 0 to 255.

That's why the starting vocabulary is exactly 256. It covers every possible byte, so the tokenizer can handle any input, even text very different from its training data.

### 2. Training (BPE)

- Count every adjacent pair of tokens
- Find the most frequent pair
- Merge it into one new token (ID 256, then 257, and so on)
- Repeat

I ran [100] merges on Shakespeare (`input.txt`), giving a vocabulary of [256 + 100] tokens.

### 3. Encode and decode

- **Decode:** look up each ID's bytes, join them, decode as UTF-8 (`errors="replace"`, because a random list of IDs may not be valid UTF-8).
- **Encode:** convert text to bytes, then repeatedly apply the earliest-learned merge that applies, until none do. Using training order reproduces how the vocabulary was built.

---

## Results

- Compression on training text: **[X]x** (bytes to tokens)
- Round trip `decode(encode(s)) == s` passes for English, accented text and emoji
- Examples of what it learned: [paste 3-4 merges, e.g. `' th'`, `'e '`, `'ing'`]

| Text | UTF-8 bytes | Mine | GPT-2 | GPT-4 (cl100k) |
|---|---|---|---|---|
| English | [53] | [36] | [12] | [10] |
| Hindi | [110] | [110] | [69] | [46] |
| Telugu | [83] | [83] | [83] | [54] |

My tokenizer uses more tokens because it has [100] merges, versus about 50,000 for GPT-2 and about 100,000 for GPT-4. Hindi and Telugu cost more tokens everywhere. Each of those characters is 3 bytes in UTF-8, and tokenizers have fewer merges for them.

---

## Key Learnings

- UTF-8 is why the base vocabulary is 256. Every possible text reduces to bytes.
- Vocabulary size is a tradeoff. A bigger vocabulary gives shorter sequences, but a bigger embedding table.
- Tokenization is a source of odd LLM behaviour (spelling, counting letters), because the model sees chunks, not characters.

---

## Understood, Not Built

- Regex pre-splitting (GPT-2 / GPT-4): stops merges crossing word and punctuation boundaries
- Special tokens like `<|endoftext|>`: reserved IDs handled outside BPE
- SentencePiece / Llama tokenizers

---

## How to Run

```bash
git clone https://github.com/Abhilash-Bogavalli/Karpathy-Zero-to-Hero.git
cd tokenizer
pip install tiktoken
jupyter notebook bpe_tokenizer.ipynb
```

Needs `input.txt` (tinyshakespeare) in the same folder.

---

## References

- [Karpathy: Let's build the GPT Tokenizer](https://youtube.com/...)
- [tiktoken](https://github.com/openai/tiktoken)

---

## Author

**Abhilash Bogavalli**
[GitHub](https://github.com/Abhilash-Bogavalli) 