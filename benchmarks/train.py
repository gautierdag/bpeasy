import json
import pytest
import logging
import sys
import itertools
import glob
import dataclasses
from pathlib import Path

import tokenizers
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

import bpeasy

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@dataclasses.dataclass
class TrainBPETokenizerArgs:
    datasets: str = "./benchmarks/data"

    num_characters: int = 1000
    vocab_size: int = 1024
    max_sentencepiece_length: int = 32
    normalization_rule_name: str = "gpt"

    def __post_init__(self):
        datasets = self.datasets.split(",")
        for ckpt in datasets:
            checkpoint_dir = Path(ckpt)
            assert checkpoint_dir.is_dir(), checkpoint_dir

        assert self.normalization_rule_name in [
            "gpt",
            "gpt-num2",
            "punct",
            "punct-num2",
            "identity",
        ]


def get_content_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().rstrip()
        try:
            x = json.loads(line)
        except UnicodeDecodeError as e:
            logging.info(f"Error when trying to decode '{line}': {str(e)}")
            raise
        for k in ["text", "content"]:
            if k in x:
                return k
        raise RuntimeError(f"Unable to determine key for {path}")


def jsonl_content_iterator(
    file_path: str,
    character_limit=2_000_000,
):
    """
    Iterates over a jsonl file and yields the content of each line
    Tracks the number of characters yielded and stops when the limit is reached
    This is ripe for optimisation if you want to mess with more fine-grained
    character limits (eg. more Python than Java)
    """
    logging.info(f"Creating iterator for {character_limit} characters in {file_path}")
    chunk_num, character_count = 0, 0
    chunks = glob.glob(f"{file_path}/*.jsonl")
    logging.info(f"Found {len(chunks)} chunks")

    while character_count < character_limit and chunk_num < len(chunks):
        file_name = chunks[chunk_num]
        content_key = get_content_key(file_name)
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                if character_count >= character_limit:  # stop after limit
                    break
                try:
                    obj = json.loads(line)
                    text = obj[content_key]
                except:
                    continue
                text_character_count = len(text)
                character_count += text_character_count
                yield text
        chunk_num += 1


def mix_jsonl_content_iterator(args: TrainBPETokenizerArgs):
    datasets = []
    num_datasets = len(args.datasets.split(","))
    for dataset in args.datasets.split(","):
        datasets.append((dataset, args.code_percentage / num_datasets))

    # Create iterators
    iterators = []
    total_weight = sum([t[1] for t in datasets])
    for file_path, percentage in datasets:
        effective_limit = int((percentage / total_weight) * args.num_characters)
        assert effective_limit > 0
        it = jsonl_content_iterator(
            file_path,
            effective_limit,
        )
        iterators.append(it)

    # Chain iterators together
    return itertools.chain(*iterators)


def get_regex_from_normalization_rule_name(normalization_rule_name: str) -> str:
    # GPT4 regex
    if normalization_rule_name == "gpt":
        return r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # limits to 2 digits (use for vocab size < 50k)
    elif normalization_rule_name == "gpt-num2":
        return r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # separates punctuation from words (except spaces)
    elif normalization_rule_name == "punct":
        return r""" ?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # limits to 2 digits (use for vocab size < 50k)
    elif normalization_rule_name == "punct-num2":
        return r""" ?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    else:
        raise ValueError(f"Unknown normalization_rule_name {normalization_rule_name}")


@pytest.fixture(scope="session")
def args() -> str:
    return TrainBPETokenizerArgs()


def test_train_huggingface(benchmark, args: TrainBPETokenizerArgs):
    # should be at least 0.14.0 to train with char limit
    assert tokenizers.__version__ >= "0.14.0"
    tokenizer = Tokenizer(BPE(byte_fallback=True))
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        show_progress=True,
        special_tokens=[f"<0x{i:02X}>" for i in range(256)],  # seed sm vocab
        max_token_length=args.max_sentencepiece_length,
    )
    regex_expression = get_regex_from_normalization_rule_name(
        args.normalization_rule_name
    )
    gpt_regex = Regex(regex_expression)

    split_pre_tokenizer = pre_tokenizers.Split(
        gpt_regex, behavior="isolated", invert=False
    )
    byte_pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [split_pre_tokenizer, byte_pre_tokenizer]
    )
    # Use ByteLevel Decoder
    tokenizer.decoder = decoders.Sequence(
        [decoders.ByteLevel(), decoders.ByteFallback()]
    )
    iterator = mix_jsonl_content_iterator(args)
    # training the tokenizer
    benchmark(
        tokenizer.train_from_iterator,
        iterator,
        trainer,
    )


def test_train_bpeasy(benchmark, args: TrainBPETokenizerArgs):
    # Use ByteLevel Decoder
    iterator = mix_jsonl_content_iterator(args)
    # training the tokenizer
    regex = get_regex_from_normalization_rule_name(args.normalization_rule_name)
    benchmark(
        bpeasy.train_bpe,
        iterator,
        regex,
        args.max_sentencepiece_length,
        args.vocab_size,
    )
