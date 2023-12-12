import json
import argparse

import logging
import sys
import itertools
import glob
import dataclasses
from dataclasses import asdict
from pathlib import Path
import os

import bpeasy


os.environ["TIKTOKEN_CACHE_DIR"] = ""
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

SPECIAL_TOKENS = {
    "bos_piece": "<|begin_of_text|>",
    "eos_piece": "<|end_of_text|>",
    "pad_piece": "<pad>",
    "fim_prefix": "<|fim_prefix|>",
    "fim_middle": "<|fim_middle|>",
    "fim_suffix": "<|fim_suffix|>",
}


@dataclasses.dataclass
class TrainBPETokenizerArgs:
    output_dir: str

    english_datasets: str = (
        "/Users/gautier/Github/tokenizer-benchmarks/data/english/test"
    )
    code_datasets: str = "/Users/gautier/Github/tokenizer-benchmarks/data/code/test"
    multilingual_datasets: str = (
        "/Users/gautier/Github/tokenizer-benchmarks/data/multilingual/test"
    )
    num_characters: int = 1_000_000_000
    vocab_size: int = 64_000
    max_sentencepiece_length: int = 128
    normalization_rule_name: str = "gpt"
    code_percentage: float = 0.1
    multilingual_percentage: float = 0.1

    def __post_init__(self):
        datasets = (
            self.english_datasets.split(",")
            + self.code_datasets.split(",")
            + self.multilingual_datasets.split(",")
        )
        for ckpt in datasets:
            checkpoint_dir = Path(ckpt)
            assert checkpoint_dir.is_dir(), checkpoint_dir

        assert self.code_percentage + self.multilingual_percentage <= 1
        assert self.normalization_rule_name in [
            "gpt",
            "gpt-num2",
            "punct",
            "punct-num2",
            "identity",
        ]


@dataclasses.dataclass
class ConvertBPETokenizerArgs:
    tokenizer_path: str
    reduce_size: bool = False
    normalization_rule_name: str = "gpt"


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
    code_datasets = args.code_datasets.split(",")
    mp_datasets = args.multilingual_datasets.split(",")
    en_datasets = args.english_datasets.split(",")
    for dataset in code_datasets:
        if args.code_percentage > 0:
            datasets.append((dataset, args.code_percentage / len(code_datasets)))

    for dataset in mp_datasets:
        if args.multilingual_percentage > 0:
            datasets.append((dataset, args.multilingual_percentage / len(mp_datasets)))
    for dataset in en_datasets:
        if (1 - args.code_percentage - args.multilingual_percentage) > 0:
            datasets.append(
                (
                    dataset,
                    (1 - args.code_percentage - args.multilingual_percentage)
                    / len(en_datasets),
                )
            )

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


def convert_byte_string(byte_string: bytes) -> bytes:
    # Strip the angle brackets and '0x'
    hex_value = byte_string.decode().replace("<", "").replace(">", "").replace("0x", "")
    # Convert the hex value to an integer
    int_value = int(hex_value, 16)
    # Convert the integer to a character and then to a byte string
    converted_byte_string = bytes([int_value])
    return converted_byte_string


def generate_model_name(config_dict: dict) -> str:
    param_abbreviations = {
        "vocab_size": "vs",
        "max_sentencepiece_length": "msl",
        "normalization_rule_name": "nrn",
        "num_characters": "nc",
        "code_percentage": "cp",
        "multilingual_percentage": "mp",
    }
    sorted_dict = dict(sorted(param_abbreviations.items()))
    model_name = ""
    for param, abbr in sorted_dict.items():
        try:
            model_name += abbr + "_" + str(config_dict[param]) + "_"
        except KeyError:
            logging.info(f"Skipping {param} in model name")
            continue
    model_name = model_name.rstrip("_")
    return model_name


def train(args: TrainBPETokenizerArgs):
    # Use ByteLevel Decoder
    iterator = mix_jsonl_content_iterator(args)
    # training the tokenizer
    regex = get_regex_from_normalization_rule_name(args.normalization_rule_name)
    import time
    time_now = time.time()
    vocab = bpeasy.train_bpe(
        iterator,
        regex,
        args.max_sentencepiece_length,
        args.vocab_size,
    )
    logging.info(f"Training took {time.time() - time_now} seconds")

    name = generate_model_name(asdict(args))

    bpeasy.save_vocab_to_tiktoken(
        vocab=vocab,
        out_path=args.output_dir + f"/{name}.bpeasy.tiktoken",
        special_tokens=list(SPECIAL_TOKENS.values()),
        fill_to_nearest_multiple_of_eight=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer")
    subparsers = parser.add_subparsers(dest="command")
    parser_train = subparsers.add_parser("train")

    parser_train.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The directory to save the tokenizer to",
    )
    parser_train.add_argument(
        "--english-datasets",
        type=str,
        default="data/english/train",
    )
    parser_train.add_argument(
        "--code-datasets",
        type=str,
        default="data/code/train",
    )
    parser_train.add_argument(
        "--multilingual-datasets",
        type=str,
        default="data/multilingual/train",
    )
    parser_train.add_argument(
        "--num-characters",
        type=int,
        default=1_000_000_000,
        help="The number of characters to train on",
    )
    parser_train.add_argument(
        "--vocab-size",
        type=int,
        default=64_000,
        help="The number of characters to train on",
    )
    parser_train.add_argument(
        "--max-sentencepiece-length",
        type=int,
        default=128,
        help="The maximum length of a token",
    )
    parser_train.add_argument(
        "--normalization-rule-name",
        type=str,
        default="gpt",
        help="The normalization rule to use",
    )
    parser_train.add_argument(
        "--code-percentage",
        type=float,
        default=0.1,
        help="The percentage of code to use",
    )
    parser_train.add_argument(
        "--multilingual-percentage",
        type=float,
        default=0.1,
        help="The percentage of multilingual to use",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_cfg = TrainBPETokenizerArgs(
            output_dir=args.output_dir,
            # english_datasets=args.english_datasets,
            # code_datasets=args.code_datasets,
            # multilingual_datasets=args.multilingual_datasets,
            num_characters=args.num_characters,
            vocab_size=args.vocab_size,
            max_sentencepiece_length=args.max_sentencepiece_length,
            normalization_rule_name=args.normalization_rule_name,
            code_percentage=args.code_percentage,
            multilingual_percentage=args.multilingual_percentage,
        )
        logging.info(
            f"Training with config:\n{json.dumps(asdict(train_cfg), indent=2)}"
        )
        train(train_cfg)
    else:
        raise ValueError(f"Unknown command {args.command}")
