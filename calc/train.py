"""Entrypoint shim for backward compatibility with calc.train."""

from calc.core.train import inspect_step, main

__all__ = ["inspect_step", "main"]

if __name__ == "__main__":
    main()
