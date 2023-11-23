from bpeasy import train_bpe


def test_train_bpe_vocab_size():
    vocab_size = 300
    max_token_length = 4
    regex = r"([^\s]+)|(\s+)"
    vocab = train_bpe(
        iter(["This is a test", "this is another test", "good tests"]),
        regex,
        max_token_length,
        vocab_size,
    )
    assert len(vocab) == 266


def test_train_bpe_max_token_length():
    vocab_size = 300
    max_token_length = 2
    regex = r"([^\s]+)|(\s+)"
    vocab = train_bpe(
        iter(["This is a test", "this is another test", "good tests"]),
        regex,
        max_token_length,
        vocab_size,
    )
    for token in vocab:
        assert len(token) <= max_token_length
    max_token_length = 3
    vocab = train_bpe(
        iter(["This is a test", "this is another test", "good tests"]),
        regex,
        max_token_length,
        vocab_size,
    )
    for token in vocab:
        assert len(token) <= max_token_length


def test_train_bpe_gpt_regex():
    vocab_size = 300
    max_token_length = 128
    regex = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    vocab = train_bpe(
        iter(["We've got a test", "We've got good test", "this is a good tests"]),
        regex,
        max_token_length,
        vocab_size,
    )
    for token in vocab:
        assert len(token) <= max_token_length

    print(vocab)
    assert b" go" in vocab.keys()
    assert b"'ve" in vocab.keys()
