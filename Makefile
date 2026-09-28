UV ?= uv

.PHONY: help install check train sample clean

help:
	@printf '%s\n' 'install  Install the project environment' 'check    Run tests and compile sources' 'train    Train the tiny model' 'sample   Generate text from a trained checkpoint' 'clean    Remove local artifacts'

install:
	$(UV) sync

check:
	$(UV) run python -m compileall -q calc tests
	$(UV) run python -m unittest discover -s tests

train:
	$(UV) run python -m calc.train

sample:
	$(UV) run python -m calc.sample

clean:
	rm -rf .venv __pycache__ calc/__pycache__ tests/__pycache__ artifacts
