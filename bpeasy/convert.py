import tiktoken

from typing import Optional
from functools import lru_cache
import json


# Adapted from https://github.com/openai/tiktoken/issues/60#issuecomment-1499977960
def bpe(
    mergeable_ranks: dict[bytes, int], token: bytes, max_rank: Optional[int] = None
) -> list[bytes]:
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2 :]
        )
    return parts


# Source taken from https://github.com/huggingface/transformers/blob/73de5108e172112bc620cfc0ceebfd27730dba11/src/transformers/models/gpt2/tokenization_gpt2.py#L63
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def convert_tiktoken_to_huggingface(
    encoder: tiktoken.Encoding,
    out_path: str,
    regex_pattern: str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+""",
):
    byte_encoder = bytes_to_unicode()

    def token_bytes_to_string(b):
        return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

    def generate_vocab_and_merges(encoder):
        mergeable_ranks = encoder._mergeable_ranks

        merges = []
        vocab = {}
        i = 0
        for token, rank in mergeable_ranks.items():
            vocab[token_bytes_to_string(token)] = rank

            i += 1
            if len(token) == 1:
                continue
            merged = tuple(bpe(mergeable_ranks, token, max_rank=rank))
            assert len(merged) == 2
            merges.append(" ".join(map(token_bytes_to_string, merged)))

        # Also add special tokens
        vocab.update(encoder._special_tokens)

        return vocab, merges

    vocab, merges = generate_vocab_and_merges(encoder)

    added_tokens = [
        {
            "id": id,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for content, id in encoder._special_tokens.items()
    ]

    tokenizer_template = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": regex_pattern},
                    "behavior": "Removed",
                    "invert": True,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(tokenizer_template, fp, indent=2, ensure_ascii=False)
