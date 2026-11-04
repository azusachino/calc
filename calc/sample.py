"""Entrypoint shim for backward compatibility with calc.sample."""

from calc.core.sample import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
