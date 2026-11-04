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

## Continuous integration

GitHub Actions runs `make check` for pull requests and pushes to `main`. The workflow uses `mise.toml` to install the project's pinned Python and uv versions, then verifies the locked environment before testing.

## Exploratory exercises

The repository includes standalone inspection scripts and component exercises for studying parts of the language model pipeline in isolation. See [`docs/`](docs/index.md) for full walkthroughs:

- [Tokenizers and datasets](docs/tokenizers.md) — word splitting, OOV handling, and GPT-2 BPE (`uv run calc/exercises/compare_tokenizers.py`).
- [Positional embeddings](docs/embeddings.md) — learned lookup tables, sinusoidal waveforms, and RoPE (`uv run calc/exercises/compare_embeddings.py`).

## Project map

- `calc/core/` — decoder-only Transformer model, embeddings, tokenizer, dataset, and training runtime.
- `calc/math/` — theory labs and mathematical geometry benchmarks corresponding to the *math-for-ai* curriculum.
- `calc/exercises/` — standalone inspection and comparative scripts.
- `docs/` — component exercises and exploration index.
- `tests/` — deterministic checks across core model and math modules (`make check`).
- `packages/` — optional workspace members for independent experiments and extensions.

This is a learning project. The tiny character vocabulary, small corpus, and CPU-sized model are deliberate limits; scaling and production inference are outside the starter scope.

## Baseline to compare against

On a MacBook Air M2 (24 GB unified memory), the default configuration (seed 1337, 1,000 steps, batch size 32, context length 64, 46-character vocabulary) ran on CPU in about 17.4 seconds wall time on 2026-09-29. The final sampled training loss was 1.995 and validation loss 2.395. `make sample` generated a noisy character-level continuation from the starter corpus. This is a learning baseline, not a benchmark; timings and exact numbers can vary with PyTorch version and hardware. Re-run `make train` and `make sample` before comparing a change, and record the environment and settings.

## Compare with a pretrained model

After working through the from-scratch path, use a separate inference runtime to see the same broad ideas inside a real pretrained model. Hugging Face is the model hub and model-card source; [`llama.cpp`](https://github.com/ggml-org/llama.cpp) runs compatible quantized **GGUF** models locally. They complement calc rather than replace it: calc exposes training and architecture; llama.cpp is a practical inference exercise, not a full training recipe.

For a small-model starting point, compare [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it), checking each model card, license, and current GGUF availability first. Model rankings change; see the [Hugging Face trending list](https://huggingface.co/models?sort=trending). On Apple Silicon, start with a modest quantized model and context, monitor shared memory, and check the current [llama.cpp model documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md).

No runtime or model download is part of this repository's starter setup. Keep external model files and runtime configuration outside calc's training artifacts; if adding a runnable integration later, first choose its toolchain, model/license, and testable learning goal.
