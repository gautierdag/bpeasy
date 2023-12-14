import dataclasses
import glob
import json
import logging
import sys
import time
from pathlib import Path

import tokenizers
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

import bpeasy
from bpeasy.tokenizer import BPEasyTokenizer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@dataclasses.dataclass
class TrainBPETokenizerArgs:
    dataset: str = "./benchmarks/data"
    vocab_size: int = 32_000
    max_sentencepiece_length: int = 128
    regex_pattern: str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    def __post_init__(self):
        checkpoint_dir = Path(self.dataset)
        assert checkpoint_dir.is_dir(), checkpoint_dir


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
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["text"]
                text_character_count = len(text)
                character_count += text_character_count
                yield text
        chunk_num += 1


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
    gpt_regex = Regex(args.regex_pattern)

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
    tokenizer.train_from_iterator(iterator, trainer)

    return tokenizer


def train_bpeasy(args: TrainBPETokenizerArgs):
    # Use ByteLevel Decoder
    iterator = jsonl_content_iterator(args)
    # training the tokenizer
    vocab = bpeasy.train_bpe(
        iterator,
        args.regex_pattern,
        args.max_sentencepiece_length,
        args.vocab_size,
    )

    return BPEasyTokenizer(
        vocab,
        args.regex_pattern,
        special_tokens=[],
        fill_to_nearest_multiple_of_eight=False,
    )


def encode(tokenizer, args) -> float:
    iterator = jsonl_content_iterator(args)
    lengths = []
    num_bytes = 0
    for text in iterator:
        num_bytes += len(text.encode("utf-8"))
        encoded = tokenizer.encode(text)
        lengths.append(len(encoded))
    return num_bytes / sum(lengths)


def get_mean_std_dev(times: list[float]) -> tuple[float, float]:
    avg_time = sum(times) / len(times)
    std_dev = sum([(t - avg_time) ** 2 for t in times])
    return avg_time, std_dev


if __name__ == "__main__":
    args = TrainBPETokenizerArgs()

    times_train_huggingface = []
    times_encode_huggingface = []
    times_train_bpeasy = []
    times_encode_bpeasy = []
    byte_per_token_bpeasy = []

    for v in tqdm(range(5000, 100_000, 5000)):
        args.vocab_size = v

        time_now = time.time()
        tokenizer = train_huggingface(args)
        times_train_huggingface.append(time.time() - time_now)

        time_now = time.time()
        byte_per_token_hf = encode(tokenizer, args)
        times_encode_huggingface.append(time.time() - time_now)

        time_now = time.time()
        tokenizer = train_bpeasy(args)
        times_train_bpeasy.append(time.time() - time_now)

        time_now = time.time()
        byte_per_token_bpeasy.append(encode(tokenizer, args) / byte_per_token_hf)
        times_encode_bpeasy.append(time.time() - time_now)

    m_hf, std_hf = get_mean_std_dev(times_train_huggingface)
    m_bpeasy, std_bpeasy = get_mean_std_dev(times_train_bpeasy)

    print(f"huggingface train time {m_hf} +/- {std_hf}")
    print(f"bpeasy train time {m_bpeasy} +/- {std_bpeasy}")

    m_hf, std_hf = get_mean_std_dev(times_encode_huggingface)
    m_bpeasy, std_bpeasy = get_mean_std_dev(times_encode_bpeasy)

    print(f"huggingface encode time {m_hf} +/- {std_hf}")
    print(f"bpeasy encode time {m_bpeasy} +/- {std_bpeasy}")

    m_bpeasy, std_bpeasy = get_mean_std_dev(byte_per_token_bpeasy)
    print(f"bpeasy bytes/token vs hf: {m_bpeasy} +/- {std_bpeasy}")
