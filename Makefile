UV ?= uv

.PHONY: help install check train inspect compare-tokenizers compare-embeddings sample clean

help:
	@printf '%s\n' 'install  Install the project environment' 'check    Run tests and compile sources' 'inspect  Examine one training step' 'compare-tokenizers Compare tokenizer counts and next-token windows' 'compare-embeddings Compare embedding lookups and positional encodings' 'train    Train the tiny model' 'sample   Generate text from a trained checkpoint' 'clean    Remove local artifacts'

install:
	$(UV) sync

check:
	$(UV) run python -m compileall -q calc tests
	$(UV) run python -m unittest discover -s tests

inspect:
	$(UV) run python -m calc.train --inspect-step

compare-tokenizers:
	$(UV) run python -m calc.compare_tokenizers

compare-embeddings:
	$(UV) run python -m calc.compare_embeddings

train:
	$(UV) run python -m calc.train

sample:
	$(UV) run python -m calc.sample

clean:
	rm -rf .venv __pycache__ calc/__pycache__ tests/__pycache__ artifacts
