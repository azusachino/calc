import unittest
from unittest.mock import Mock, patch

from calc.tokenizer import (
    CharTokenizer,
    GPT2BPETokenizer,
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocab,
)


class CharTokenizerTests(unittest.TestCase):
    def test_round_trip(self):
        tokenizer = CharTokenizer("hello")
        self.assertEqual(tokenizer.decode(tokenizer.encode("hello")), "hello")

    def test_unknown_character_is_clear(self):
        tokenizer = CharTokenizer("abc")
        with self.assertRaisesRegex(ValueError, "unknown character"):
            tokenizer.encode("z")


class GPT2BPETokenizerTests(unittest.TestCase):
    @patch("calc.tokenizer.tiktoken.get_encoding")
    def test_uses_gpt2_encoding_and_allows_end_of_text_marker(self, get_encoding):
        encoding = Mock(max_token_value=50256)
        encoding.encode.return_value = [50256]
        encoding.decode.return_value = "<|endoftext|>"
        get_encoding.return_value = encoding

        tokenizer = GPT2BPETokenizer()

        self.assertEqual(tokenizer.vocab_size, 50257)
        self.assertEqual(tokenizer.encode("<|endoftext|>"), [50256])
        self.assertEqual(tokenizer.decode([50256]), "<|endoftext|>")
        get_encoding.assert_called_once_with("gpt2")
        encoding.encode.assert_called_once_with(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )
        encoding.decode.assert_called_once_with([50256])


class SimpleTokenizerTests(unittest.TestCase):
    def test_v1_encodes_and_decodes_known_words_and_punctuation(self):
        tokenizer = SimpleTokenizerV1(build_vocab("Hello, world!"))
        self.assertEqual(tokenizer.decode(tokenizer.encode("Hello, world!")), "Hello, world!")

    def test_v1_rejects_out_of_vocabulary_tokens(self):
        tokenizer = SimpleTokenizerV1(build_vocab("hello world"))
        with self.assertRaises(KeyError):
            tokenizer.encode("new word")

    def test_v2_maps_unknown_tokens_to_unk(self):
        vocab = build_vocab("hello world", special_tokens=("<|endoftext|>", "<|unk|>"))
        tokenizer = SimpleTokenizerV2(vocab)
        ids = tokenizer.encode("hello stranger")
        self.assertEqual(tokenizer.decode(ids), "hello <|unk|>")

    def test_v2_requires_unk_token(self):
        with self.assertRaisesRegex(ValueError, "<\\|unk\\|>"):
            SimpleTokenizerV2(build_vocab("hello world"))


if __name__ == "__main__":
    unittest.main()
