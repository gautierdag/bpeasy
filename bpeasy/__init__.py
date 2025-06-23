from importlib.metadata import version

from .bpeasy import train_bpe_stream

__version__ = version("bpeasy")

def train_bpe(
    iterator,
    python_regex: str,
    max_token_length: int,
    vocab_size: int,
    batch_size: int = 1000,
) -> dict[bytes, int]:
    """
    Train a BPE tokenizer on the given iterator of strings.
    
    Args:
        iterator: An iterator over strings to train the BPE tokenizer on.
        python_regex: A regex pattern to use for tokenization.
        max_token_length: The maximum length of tokens.
        vocab_size: The desired vocabulary size.
        batch_size (default: 1000): The number of samples to process in each batch (how many iterator items to process at once).

    Returns a vocabulary mapping bytes to integer ranks.
    """
    return train_bpe_stream(iterator, python_regex, max_token_length, vocab_size, batch_size)


def save_vocab_to_tiktoken(
    vocab: dict[bytes, int],
    out_path: str,
    special_tokens: list[str] = [],
    fill_to_nearest_multiple_of_eight: bool = False,
) -> None:
    """
    Export vocab to tiktoken txt format - use this if you want to use tiktoken library directly
    Note: you will need to handle special tokens and regex yourself
    """
    import base64

    sorted_vocab = sorted(list(vocab.items()), key=lambda x: x[1])
    for special_token in special_tokens:
        sorted_vocab.append((special_token.encode("utf-8"), len(sorted_vocab)))

    if fill_to_nearest_multiple_of_eight:
        while len(sorted_vocab) % 8 != 0:
            sorted_vocab.append(
                (f"<|special-{len(sorted_vocab)}|>".encode("utf-8"), len(sorted_vocab))
            )

    with open(out_path, "wb") as f:
        for token, rank in sorted_vocab:
            # encode token to base64 and write to file with rank separated by a space
            f.write(base64.b64encode(token) + b" " + str(rank).encode("utf-8") + b"\n")


__all__ = [
    "save_vocab_to_tiktoken",
    "train_bpe",
    "train_bpe_stream",
    "__version__",
]