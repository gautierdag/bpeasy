import base64
from unittest.mock import mock_open, patch, call

from bpeasy import save_vocab_to_tiktoken


def test_basic_functionality():
    vocab = {b"hello": 1, b"world": 2}
    with patch("builtins.open", mock_open()) as mock_file:
        save_vocab_to_tiktoken(vocab, "path/to/file.txt")
        mock_file.assert_called_once_with("path/to/file.txt", "wb")
        # Check if the sorted vocab is written correctly
        expected_content = [
            base64.b64encode(b"hello") + b" 1\n",
            base64.b64encode(b"world") + b" 2\n",
        ]
        mock_file().write.assert_has_calls(
            [call(content) for content in expected_content]
        )


def test_special_tokens_addition():
    vocab = {b"token": 0}
    special_tokens = ["special1", "special2"]
    with patch("builtins.open", mock_open()) as mock_file:
        save_vocab_to_tiktoken(vocab, "path/to/file.txt", special_tokens)
        # Check if special tokens are added correctly
        expected_content = [
            base64.b64encode(b"token") + b" 0\n",
            base64.b64encode("special1".encode("utf-8")) + b" 1\n",
            base64.b64encode("special2".encode("utf-8")) + b" 2\n",
        ]
        mock_file().write.assert_has_calls(
            [call(content) for content in expected_content]
        )


def test_fill_to_nearest_multiple_of_eight():
    vocab = {b"token": 0}
    with patch("builtins.open", mock_open()) as mock_file:
        save_vocab_to_tiktoken(
            vocab, "path/to/file.txt", fill_to_nearest_multiple_of_eight=True
        )
        # Verify that additional tokens are added to make the count a multiple of eight
        mock_file().write.assert_called()
        # Check the exact content based on your logic of filling to nearest multiple of eight
        assert mock_file().write.call_count == 8
