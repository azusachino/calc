# calc documentation and exercise index

`calc` is a hands-on learning project for building a tiny autoregressive language model from scratch.

In addition to the core character-level model (`calc/model.py`, `calc/train.py`, `calc/sample.py`), the codebase includes standalone, inspectable modules and exercises for exploring individual components of modern LLMs.

## Exploratory exercises

Run any exercise directly with `uv run`:

| Topic | Script / Module | Command | Documentation |
| --- | --- | --- | --- |
| **Vector Geometry & High-Dim Spaces** | `calc/vector_geometry.py` | `uv run calc/vector_geometry.py` | [Vector Geometry](geometry.md) |
| **Dot Products & Scaled Attention** | `calc/attention_math.py` | `uv run calc/attention_math.py` | [Attention Mathematics](attention_math.md) |
| **Linear Transformations & QKV** | `calc/linear_transformations.py` | `uv run calc/linear_transformations.py` | [Transformations and QKV](transformations.md) |
| **Rotations & RoPE Relative Math** | `calc/rotations_rope.py` | `uv run calc/rotations_rope.py` | [Rotations and RoPE](rotations.md) |
| **Gradients & Loss Surfaces** | `calc/gradient_descent.py` | `uv run calc/gradient_descent.py` | [Gradients and Backprop](gradients.md) |
| **Probability & Discrete Sampling** | `calc/sampling_math.py` | `uv run calc/sampling_math.py` | [Probability and Sampling](sampling.md) |
| **Tokenizers & Datasets** | `calc/compare_tokenizers.py`, `calc/dataset.py` | `uv run calc/compare_tokenizers.py` | [Tokenizers and Datasets](tokenizers.md) |
| **Positional Encodings** | `calc/compare_embeddings.py`, `calc/embeddings.py` | `uv run calc/compare_embeddings.py` | [Positional Embeddings](embeddings.md) |
| **Single-Step Training Probe** | `calc/train.py` | `make inspect` (or `uv run calc/train.py --inspect-step`) | [README Quick Start](../README.md#quick-start) |

## Core model and training

- See [README.md](../README.md) for the project map, baseline training run, and quick start commands.
- See `tests/` for deterministic verification checks across all modules (`make check`).
