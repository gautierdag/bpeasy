import pytest
import bpeasy


def test_train():
    basic_iterator = iter(["This is some texts", "this is more texts", "cool texts"])
    vocab_size = 10
    max_token_length = 4
    regex = r"([^\s]+)|(\s+)"

    vocab = bpeasy.train_bpe(basic_iterator, regex, max_token_length, vocab_size)

    print(len(vocab))
    print(vocab)
