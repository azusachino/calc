"""Compare token counts and next-token windows across tokenizer exercises."""

from pathlib import Path

from calc.core.tokenizer import (
    CharTokenizer,
    GPT2BPETokenizer,
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocab,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "input.txt"


def main() -> None:
    text = DATA.read_text(encoding="utf-8")
    char = CharTokenizer(text)
    vocab = build_vocab(text)
    v1 = SimpleTokenizerV1(vocab)
    v2_vocab = build_vocab(text, ("<|endoftext|>", "<|unk|>"))
    v2 = SimpleTokenizerV2(v2_vocab)
    bpe = GPT2BPETokenizer()
    tokenizers = [
        ("character", char, char.vocab_size),
        ("SimpleTokenizer V1", v1, len(vocab)),
        ("SimpleTokenizer V2", v2, len(v2_vocab)),
    ]
    tokenizers.append(("GPT-2 BPE", bpe, bpe.vocab_size))

    print(f"{'Tokenizer':<20} {'vocabulary':>10} {'corpus tokens':>14}")
    for name, tokenizer, vocab_size in tokenizers:
        ids = tokenizer.encode(text)
        print(f"{name:<20} {vocab_size:>10} {len(ids):>14}")
        print(f"  input:  {tokenizer.decode(ids[:8])!r}")
        print(f"  target: {tokenizer.decode(ids[1:9])!r}")

    unseen = "unseenwordzz"
    try:
        v1.encode(unseen)
    except KeyError:
        print("\nV1 unseen token: KeyError")
    print(f"V2 unseen token: {v2.decode(v2.encode(unseen))}")


if __name__ == "__main__":
    main()
