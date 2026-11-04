import unittest

from calc.core.dataset import GPTDatasetV1


class StubTokenizer:
    def encode(self, text):
        return [int(token) for token in text.split()]


class GPTDatasetTests(unittest.TestCase):
    def test_sliding_windows_pair_inputs_with_next_tokens(self):
        dataset = GPTDatasetV1("0 1 2 3 4 5 6 7", StubTokenizer(), max_length=4, stride=2)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0][0].tolist(), [0, 1, 2, 3])
        self.assertEqual(dataset[0][1].tolist(), [1, 2, 3, 4])
        self.assertEqual(dataset[1][0].tolist(), [2, 3, 4, 5])
        self.assertEqual(dataset[1][1].tolist(), [3, 4, 5, 6])

    def test_rejects_invalid_window_or_short_text(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            GPTDatasetV1("0 1 2", StubTokenizer(), max_length=0, stride=1)
        with self.assertRaisesRegex(ValueError, "more tokens"):
            GPTDatasetV1("0 1", StubTokenizer(), max_length=2, stride=1)


if __name__ == "__main__":
    unittest.main()
