import unittest

from calc.tokenizer import CharTokenizer


class CharTokenizerTests(unittest.TestCase):
    def test_round_trip(self):
        tokenizer = CharTokenizer("hello")
        self.assertEqual(tokenizer.decode(tokenizer.encode("hello")), "hello")

    def test_unknown_character_is_clear(self):
        tokenizer = CharTokenizer("abc")
        with self.assertRaisesRegex(ValueError, "unknown character"):
            tokenizer.encode("z")


if __name__ == "__main__":
    unittest.main()
