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
make train
make sample
```

The starter corpus is `data/input.txt`. Training is intentionally CPU-friendly and produces a local checkpoint under `artifacts/` (ignored by Git). Edit the corpus or model settings in `calc/config.py` and rerun training.

## Project map

- `calc/tokenizer.py` — deterministic character vocabulary and encode/decode.
- `calc/model.py` — causal decoder-only Transformer.
- `calc/train.py` — data preparation, training loop, and checkpoint writing.
- `calc/sample.py` — load a checkpoint and generate text.
- `tests/` — small deterministic checks for the core contracts.
- `packages/` — optional workspace members for independent experiments and extensions.

This is a learning project. The tiny character vocabulary, small corpus, and CPU-sized model are deliberate limits; scaling, subword tokenization, and production inference are outside the starter scope.
