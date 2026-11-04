# calc documentation and exercise index

`calc` is a hands-on learning project for building a tiny autoregressive language model from scratch.

The codebase is organized into three clean subpackages:
- **`calc.core`** — the minimal autoregressive Transformer (`model.py`, `embeddings.py`, `tokenizer.py`, `dataset.py`, `train.py`, `sample.py`).
- **`calc.math`** — theory labs and visual geometry benchmarks corresponding to the *AI 数学与几何直觉 (`math-for-ai`)* series.
- **`calc.exercises`** — standalone comparative scripts evaluating tokenizers and positional embeddings.

## Exploratory exercises and math labs

Run any exercise or math benchmark directly with `uv run`:

| Topic | Script / Module | Command | Documentation |
| --- | --- | --- | --- |
| **Vector Geometry & High-Dim Spaces** | `calc/math/vectors.py` | `uv run calc/math/vectors.py` | [Vector Geometry](geometry.md) |
| **Dot Products & Scaled Attention** | `calc/math/attention.py` | `uv run calc/math/attention.py` | [Attention Mathematics](attention_math.md) |
| **Linear Transformations & QKV** | `calc/math/transformations.py` | `uv run calc/math/transformations.py` | [Transformations and QKV](transformations.md) |
| **Rotations & RoPE Relative Math** | `calc/math/rotations.py` | `uv run calc/math/rotations.py` | [Rotations and RoPE](rotations.md) |
| **Gradients & Loss Surfaces** | `calc/math/gradients.py` | `uv run calc/math/gradients.py` | [Gradients and Backprop](gradients.md) |
| **Probability & Discrete Sampling** | `calc/math/sampling.py` | `uv run calc/math/sampling.py` | [Probability and Sampling](sampling.md) |
| **Tokenizers & Datasets** | `calc/exercises/compare_tokenizers.py` | `uv run calc/exercises/compare_tokenizers.py` | [Tokenizers and Datasets](tokenizers.md) |
| **Positional Encodings** | `calc/exercises/compare_embeddings.py` | `uv run calc/exercises/compare_embeddings.py` | [Positional Embeddings](embeddings.md) |
| **Single-Step Training Probe** | `calc/core/train.py` | `make inspect` (or `uv run -m calc.core.train --inspect-step`) | [README Quick Start](../README.md#quick-start) |

## Core model and training

- See [README.md](../README.md) for the project map, baseline training run, and quick start commands.
- See `tests/` for deterministic verification checks across core and math modules (`make check`).
