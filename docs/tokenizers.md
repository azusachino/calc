# Tokenizers and next-token datasets

This document covers the standalone tokenizer exercises and sliding-window dataset implementations in `calc`.

## 1. Word and regex tokenizers (`SimpleTokenizerV1` and `V2`)

Following Chapter 2 of Sebastian Raschka's *Build a Large Language Model (From Scratch)*:

```python
from calc.tokenizer import SimpleTokenizerV1, SimpleTokenizerV2, build_vocab

text = "Hello, world!"
v1 = SimpleTokenizerV1(build_vocab(text))

v2_vocab = build_vocab(text, special_tokens=("<|endoftext|>", "<|unk|>"))
v2 = SimpleTokenizerV2(v2_vocab)
print(v2.decode(v2.encode("Hello, there!")))  # Hello, <|unk|>!
```

- **`SimpleTokenizerV1`**: Splits on punctuation and whitespace using regular expressions. Fails with `KeyError` on unseen tokens.
- **`SimpleTokenizerV2`**: Introduces an `<|unk|>` fallback token for out-of-vocabulary words.

## 2. Byte-Pair Encoding (`GPT2BPETokenizer`)

Uses OpenAI's `tiktoken` to demonstrate subword tokenization:

```python
from calc.tokenizer import GPT2BPETokenizer

bpe = GPT2BPETokenizer()
ids = bpe.encode("hello world")
print(ids)                  # [31373, 995]
print(bpe.decode(ids))      # hello world
print(bpe.vocab_size)       # 50257
```

## 3. Sliding-window dataset (`GPTDatasetV1`)

Encodes text with `GPT2BPETokenizer` and returns input and shifted next-token target windows:

```python
from calc.dataset import GPTDatasetV1
from calc.tokenizer import GPT2BPETokenizer

text = "The quick brown fox jumps over the lazy dog."
dataset = GPTDatasetV1(text, GPT2BPETokenizer(), max_length=4, stride=1)
inputs, targets = dataset[0]
print(inputs.tolist())
print(targets.tolist())  # same sequence shifted one token forward
```

## Runnable comparison script

Run the tokenizer comparison:

```bash
uv run calc/exercises/compare_tokenizers.py
```

Compares vocabulary size, total token count on `data/input.txt`, first next-token window, and OOV fallback handling across character-level, SimpleTokenizer, and BPE.
