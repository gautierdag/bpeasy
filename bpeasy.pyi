from typing import Iterator

def train_bpe(
    iterator: Iterator[str],
    python_regex: str,
    max_token_length: int,
    vocab_size: int,
) -> dict[bytes, int]: ...
