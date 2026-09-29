import unittest

from calc.tokenizer import (
    CharTokenizer,
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
