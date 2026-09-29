# calc — build a tiny language model from scratch

A hands-on, incremental project for understanding how language models work. We start with a character-level dataset and tokenizer, then build and train a small decoder-only Transformer with PyTorch. The goal is to make every step inspectable, not to produce a useful general-purpose model.

## Learning path

1. **Data and tokens** — turn plain text into integer token IDs and back.
2. **Causal attention** — mask future positions so predictions can only use earlier tokens.
3. **Transformer language model** — combine attention and feed-forward blocks to predict the next token.
4. **Training and sampling** — train on a tiny corpus, compare training and validation loss, and generate samples.

Each step should be runnable locally and small enough to modify. No pretrained weights or hosted model APIs are used. PyTorch supplies tensor operations and automatic differentiation; the model architecture and training loop live in this repository. The root is a uv workspace: optional experiments can be added as independent packages under `packages/` without coupling them to the core exercises.

## Quick start

Requires Python 3.12 or later and `uv`.

```sh
make install
make check
make inspect  # inspect one training step without saving a checkpoint
make train
make sample
```

The starter corpus is `data/input.txt`. Training is intentionally CPU-friendly and produces a local checkpoint under `artifacts/` (ignored by Git). Training and sampling use the seed in `calc/config.py` for comparable runs in the same software environment. Edit the corpus or model settings there and rerun training.

`make inspect` shows one sampled batch, input/target/logit shapes, decoded next-character examples, loss, and two gradient norms, then applies one optimizer step without writing the checkpoint. Each target sequence is the input shifted forward by one character, so each logit is scored against the next character. `make train` runs the full baseline.

## Tokenizer exercise: SimpleTokenizer V1 and V2

The `SimpleTokenizerV1` and `SimpleTokenizerV2` exercises follow Chapter 2 of Sebastian Raschka's [book code](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb): V1 maps regex-split words and punctuation through a supplied vocabulary; V2 adds an `<|unk|>` fallback for out-of-vocabulary pieces. Build a deterministic vocabulary with `build_vocab`:

```python
from calc.tokenizer import SimpleTokenizerV1, SimpleTokenizerV2, build_vocab

text = "Hello, world!"
v1 = SimpleTokenizerV1(build_vocab(text))

v2_vocab = build_vocab(text, special_tokens=("<|endoftext|>", "<|unk|>"))
v2 = SimpleTokenizerV2(v2_vocab)
print(v2.decode(v2.encode("Hello, there!")))  # Hello, <|unk|>!
```

These are explanatory word-token examples, not GPT-2's BPE tokenizer. The current model and training loop remain character-level and use `CharTokenizer`; these alternatives are not drop-in replacements for its checkpoints.

The next tokenizer step follows the book's [Chapter 2 GPT-2 tokenizer example](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb): use OpenAI's [`tiktoken`](https://github.com/openai/tiktoken) implementation of byte-pair encoding (BPE).

```python
from calc.tokenizer import GPT2BPETokenizer

bpe = GPT2BPETokenizer()
ids = bpe.encode("hello world")
print(ids)                  # [31373, 995]
print(bpe.decode(ids))      # hello world
print(bpe.vocab_size)       # 50257
```

The first `GPT2BPETokenizer()` call may download the GPT-2 encoding data into tiktoken's cache; it does not download model weights. Only `<|endoftext|>` is explicitly accepted as a special token. Compare how many IDs character, SimpleTokenizer, and GPT-2 BPE produce for the same text. The model and trainer still use `CharTokenizer`: switching to BPE also requires matching model vocabulary, checkpoint, and data-pipeline changes, which this exercise deliberately does not make.

## Project map

- `calc/tokenizer.py` — deterministic character tokenizer, book-inspired SimpleTokenizer V1/V2 exercises, and a GPT-2 BPE comparison using tiktoken.
- `calc/model.py` — decoder-only Transformer, including explicit Q/K/V projections and causal masking.
- `calc/train.py` — data preparation, training loop, and checkpoint writing.
- `calc/sample.py` — load a checkpoint and generate text.
- `tests/` — small deterministic checks for the core contracts.
- `packages/` — optional workspace members for independent experiments and extensions.

This is a learning project. The tiny character vocabulary, small corpus, and CPU-sized model are deliberate limits; scaling and production inference are outside the starter scope.

## Baseline to compare against

On a MacBook Air M2 (24 GB unified memory), the default configuration (seed 1337, 1,000 steps, batch size 32, context length 64, 46-character vocabulary) ran on CPU in about 17.4 seconds wall time on 2026-09-29. The final sampled training loss was 1.995 and validation loss 2.395. `make sample` generated a noisy character-level continuation from the starter corpus. This is a learning baseline, not a benchmark; timings and exact numbers can vary with PyTorch version and hardware. Re-run `make train` and `make sample` before comparing a change, and record the environment and settings.

## Compare with a pretrained model

After working through the from-scratch path, use a separate inference runtime to see the same broad ideas inside a real pretrained model. Hugging Face is the model hub and model-card source; [`llama.cpp`](https://github.com/ggml-org/llama.cpp) runs compatible quantized **GGUF** models locally. They complement calc rather than replace it: calc exposes training and architecture; llama.cpp is a practical inference exercise, not a full training recipe.

For a small-model starting point, compare [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it), checking each model card, license, and current GGUF availability first. Model rankings change; see the [Hugging Face trending list](https://huggingface.co/models?sort=trending). On Apple Silicon, start with a modest quantized model and context, and monitor shared memory. See the workstation [local models learning track](../../docs/tutorials/local-models/index.md) for model concepts and a tested llama.cpp inference run.

No runtime or model download is part of this repository's starter setup. Keep external model files and runtime configuration outside calc's training artifacts; if adding a runnable integration later, first choose its toolchain, model/license, and testable learning goal.
