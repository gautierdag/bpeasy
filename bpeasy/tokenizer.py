import json
import base64
from typing import Iterator

import tiktoken

from .bpeasy import train_bpe
from .convert import convert_tiktoken_to_huggingface


_DEFAULT_REGEX_PATTERN = r"""[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class BPEasyTokenizer:
    def __init__(
        self,
        vocab: dict[bytes, int],
        regex_pattern: str = _DEFAULT_REGEX_PATTERN,
        special_tokens: list[str] = [],
        fill_to_nearest_multiple_of_eight=False,
        name="bpeasy",
    ):
        """
        Wrapper around tiktoken.Encoding
        Handles the loading/saving of vocab/special_tokens/regex
        """

        self.name = name
        self.regex_pattern = regex_pattern
        self.special_tokens = special_tokens
        self.vocab = vocab

        # Sort the vocab by rank
        sorted_vocab = sorted(list(vocab.items()), key=lambda x: x[1])

        # add special tokens
        special_token_ranks = {}
        for special_token in special_tokens:
            special_token_ranks[special_token] = len(sorted_vocab)
            sorted_vocab.append((special_token.encode("utf-8"), len(sorted_vocab)))

        full_vocab = dict(sorted_vocab)

        # fill to nearest multiple of 8
        if fill_to_nearest_multiple_of_eight:
            while len(sorted_vocab) % 8 != 0:
                sorted_vocab.append(
                    (
                        f"<|special-{len(sorted_vocab)}|>".encode("utf-8"),
                        len(sorted_vocab),
                    )
                )

        self._encoder = tiktoken.Encoding(
            name=name,
            pat_str=self.regex_pattern,
            mergeable_ranks=full_vocab,
            special_tokens=special_token_ranks,
        )

    def encode(self, text: str, **kwargs) -> list[int]:
        return self._encoder.encode(text, **kwargs)

    def decode(self, tokens: list[int], **kwargs) -> str:
        return self._encoder.decode(tokens, **kwargs)

    @classmethod
    def from_file(cls, file_path: str) -> "BPEasyTokenizer":
        with open(file_path, "r") as file:
            data = json.load(file)
            bytes_vocab = {
                base64.b64decode(key): value for key, value in data["vocab"].items()
            }
            instance = cls(
                name=data["name"],
                vocab=bytes_vocab,
                regex_pattern=data["regex_pattern"],
                special_tokens=data["special_tokens"],
            )
            return instance

    def save(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            json.dump(
                {
                    "name": self.name,
                    "regex_pattern": self.regex_pattern,
                    "special_tokens": self.special_tokens,
                    "vocab": {
                        base64.b64encode(key).decode("utf-8"): value
                        for key, value in self.vocab.items()
                    },
                },
                file,
            )

    def export_to_huggingface_format(self, out_path: str) -> None:
        convert_tiktoken_to_huggingface(self._encoder, out_path, self.regex_pattern)

    def __len__(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(
        cls,
        iterator: Iterator[str],
        vocab_size: int = 32_000,
        max_token_length=128,
        regex_pattern: str = _DEFAULT_REGEX_PATTERN,
        special_tokens: list[str] = [],
        fill_to_nearest_multiple_of_eight=False,
        name="bpeasy",
    ) -> "BPEasyTokenizer":
        bytes_vocab = train_bpe(iterator, regex_pattern, max_token_length, vocab_size)
        return cls(
            name=name,
            vocab=bytes_vocab,
            regex_pattern=regex_pattern,
            special_tokens=special_tokens,
            fill_to_nearest_multiple_of_eight=fill_to_nearest_multiple_of_eight,
        )
