UV ?= uv

.PHONY: help install check train inspect sample clean

help:
	@printf '%s\n' 'install  Install the project environment' 'check    Run tests and compile sources' 'inspect  Examine one training step' 'train    Train the tiny model' 'sample   Generate text from a trained checkpoint' 'clean    Remove local artifacts'

install:
	$(UV) sync

check:
	$(UV) run python -m compileall -q calc tests
	$(UV) run python -m unittest discover -s tests

inspect:
	$(UV) run python -m calc.core.train --inspect-step

train:
	$(UV) run python -m calc.core.train

sample:
	$(UV) run python -m calc.core.sample

clean:
	rm -rf .venv __pycache__ calc/__pycache__ tests/__pycache__ artifacts
