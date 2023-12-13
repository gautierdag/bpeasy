import dataclasses
import glob
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import tokenizers
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

import bpeasy

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@dataclasses.dataclass
class TrainBPETokenizerArgs:
    dataset: str = "./benchmarks/data"

    vocab_size: int = 32_000
    max_sentencepiece_length: int = 64
    normalization_rule_name: str = "gpt"

    def __post_init__(self):
        checkpoint_dir = Path(self.dataset)
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
    args: TrainBPETokenizerArgs,
):
    """
    Iterates over a jsonl file and yields the content of each line
    Tracks the number of characters yielded and stops when the limit is reached
    This is ripe for optimisation if you want to mess with more fine-grained
    character limits (eg. more Python than Java)
    """
    file_path = args.dataset
    chunk_num, character_count = 0, 0
    chunks = glob.glob(f"{file_path}/*.jsonl")

    while chunk_num < len(chunks):
        file_name = chunks[chunk_num]
        content_key = get_content_key(file_name)
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj[content_key]
                except:
                    continue
                text_character_count = len(text)
                character_count += text_character_count
                yield text
        chunk_num += 1


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


def train_huggingface(args: TrainBPETokenizerArgs):
    # should be at least 0.14.0 to train with char limit
    assert tokenizers.__version__ >= "0.14.0"
    tokenizer = Tokenizer(BPE(byte_fallback=True))
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[f"<0x{i:02X}>" for i in range(256)],  # seed sm vocab
        max_token_length=args.max_sentencepiece_length,
        show_progress=False,
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
    iterator = jsonl_content_iterator(args)
    # training the tokenizer
    with suppress_stdout():
        tokenizer.train_from_iterator(iterator, trainer)


def train_bpeasy(args: TrainBPETokenizerArgs):
    # Use ByteLevel Decoder
    iterator = jsonl_content_iterator(args)
    # training the tokenizer
    regex = get_regex_from_normalization_rule_name(args.normalization_rule_name)

    bpeasy.train_bpe(
        iterator,
        regex,
        args.max_sentencepiece_length,
        args.vocab_size,
    )


if __name__ == "__main__":
    NUM_ITERATIONS = 100
    args = TrainBPETokenizerArgs()

    times_huggingface = []
    times_bpeasy = []
    for i in tqdm(range(NUM_ITERATIONS)):
        time_now = time.time()
        train_huggingface(args)
        times_huggingface.append(time.time() - time_now)

        time_now = time.time()
        train_bpeasy(args)
        times_bpeasy.append(time.time() - time_now)

    avg_time_huggingface = sum(times_huggingface) / len(times_huggingface)
    avg_time_bpeasy = sum(times_bpeasy) / len(times_bpeasy)
    std_dev_huggingface = sum(
        [(t - avg_time_huggingface) ** 2 for t in times_huggingface]
    )
    std_dev_bpeasy = sum([(t - avg_time_bpeasy) ** 2 for t in times_bpeasy])

    print(f"huggingface {avg_time_huggingface} +/- {std_dev_huggingface}")
    print(f"bpeasy {avg_time_bpeasy} +/- {std_dev_bpeasy}")
