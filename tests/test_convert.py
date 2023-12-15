from bpeasy.convert import bpe


def test_bpe_function():
    mergeable_ranks = {b"ab": 0, b"bc": 1, b"cd": 2}
    token = b"abcd"
    result = bpe(mergeable_ranks, token)
    assert result == [
        b"ab",
        b"cd",
    ], "The bpe function did not split the token correctly"
