"""Simple tokenizers for the character and book-inspired word-token exercises."""

import re

import tiktoken


_TOKEN_PATTERN = re.compile(r"([,.:;?_!\"()']|--|\s)")
_UNK = "<|unk|>"


def _split(text: str) -> list[str]:
    return [piece.strip() for piece in _TOKEN_PATTERN.split(text) if piece.strip()]


def build_vocab(text: str, special_tokens: tuple[str, ...] = ()) -> dict[str, int]:
    """Build a deterministic vocabulary, appending any special tokens last."""
    vocabulary = {token: index for index, token in enumerate(sorted(set(_split(text))))}
    for token in special_tokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    return vocabulary


class CharTokenizer:
    """A tiny deterministic character tokenizer for the training corpus."""

    def __init__(self, text: str):
        if not text:
            raise ValueError("training text must not be empty")
        self.tokens = sorted(set(text))
        self.to_id = {token: index for index, token in enumerate(self.tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.to_id[token] for token in text]
        except KeyError as error:
            raise ValueError(f"unknown character: {error.args[0]!r}") from None

    def decode(self, ids: list[int]) -> str:
        return "".join(self.tokens[index] for index in ids)


class GPT2BPETokenizer:
    """A thin wrapper around the book's pretrained GPT-2 BPE encoding."""

    def __init__(self):
        self._encoding = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self._encoding.max_token_value + 1

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: list[int]) -> str:
        return self._encoding.decode(ids)


class SimpleTokenizerV1:
    """Regex word/punctuation tokenizer; unknown tokens raise KeyError."""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {index: token for token, index in vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.str_to_int[token] for token in _split(text)]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[index] for index in ids)
        return re.sub(r"\s+([,.:;?!\"()'])", r"\1", text)


class SimpleTokenizerV2(SimpleTokenizerV1):
    """V1 with an explicit unknown-token fallback from the book's example."""

    def __init__(self, vocab: dict[str, int]):
        if _UNK not in vocab:
            raise ValueError(f'v2 vocabulary must include "{_UNK}"')
        super().__init__(vocab)
        self.unk_id = vocab[_UNK]

    def encode(self, text: str) -> list[int]:
        return [self.str_to_int.get(token, self.unk_id) for token in _split(text)]
