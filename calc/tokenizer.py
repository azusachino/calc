"""A tiny, deterministic character tokenizer for the learning corpus."""


class CharTokenizer:
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
