from importlib.metadata import version

from .bpeasy import train_bpe

__version__ = version("bpeasy")


__all__ = [
    "save_vocab_to_tiktoken",
    "train_bpe",
    "__version__",
]


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
