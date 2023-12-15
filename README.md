# bpeasy

[![codecov](https://codecov.io/gh/gautierdag/bpeasy/branch/main/graph/badge.svg?token=NWHDJ22L8I)](https://codecov.io/gh/gautierdag/bpeasy) [![tests](https://github.com/gautierdag/bpeasy/actions/workflows/test.yml/badge.svg)](https://github.com/gautierdag/bpeasy/actions/workflows/test.yml) [![image](https://img.shields.io/pypi/l/bpeasy.svg)](https://pypi.python.org/pypi/bpeasy) [![image](https://img.shields.io/pypi/pyversions/bpeasy.svg)](https://pypi.python.org/pypi/bpeasy) [![PyPI version](https://badge.fury.io/py/bpeasy.svg)](https://badge.fury.io/py/bpeasy)

## Overview

`bpeasy` is a Python package that provides a tokenizer trainer, implementing in 400 lines of rust an efficient version of Byte Pair Encoding (BPE). The implementation largely follows the huggingface `tokenizers` library, but makes opinionated decisions to simplify the tokenizer training specifically to:

1. Treat text data at the byte-level first --- all text is converted to bytes before training rather than using a character-level approach (like in Huggingface).
2. Always use a regex-based split pre-tokenizer. This is a customisable regex that is applied to the text before training. This regex decides where to split the text and limits what kind of tokens are possible. This is technically possible in Huggingface but is not well documented. We also use the `fancy-regex` crate which supports a richer set of regex features than the `regex` crate used in Huggingface.
3. Use `int64` types for counting to allow for training on much larger datasets without the risk of overflow.

You can think of `bpeasy` as the `tiktoken` training code that was never released.

## Installation

Simply install the package using pip:

```bash
pip install bpeasy
```

## Training

The training function is designed to be bare-bones and returns the trained tokenizer vocab as a dictionary of bytes to integers. This is to allow for maximum flexibility in how you want to use the tokenizer. For example, you can use then port these to tiktoken or Huggingface tokenizers (see below).

```python
# should be an iterator over str
iterator = jsonl_content_iterator(args)
# example regex from GPT-4
regex_pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# returns the vocab (dict[bytes, int])
vocab = bpeasy.train_bpe(
    iterator,
    regex_pattern,
    args.max_sentencepiece_length, # max length of tokens
    args.vocab_size, # max size of vocab
)
```

### Encoding/Decoding

To test your tokenizer you can use the `BPEasyTokenizer` class, which is a wrapper around the `tiktoken.Encoding` module, simplifying the handling of vocabularies, special tokens, and regex patterns for tokenization.

```python
from bpeasy.tokenizer import BPEasyTokenizer

your_special_tokens = ["<s>", "<pad>", "</s>"]

tokenizer = BPEasyTokenizer(
    vocab=vocab,
    regex_pattern=regex_pattern,
    special_tokens=your_special_tokens,
    fill_to_nearest_multiple_of_eight=True, # pad vocab to multiple of 8
    name="bpeasy" # optional name for the tokenizer
)

test = "hello_world"

# encode and decode uses the tiktoken functions
encoded = tokenizer.encode(test)
decoded = tokenizer.decode(encoded)
> "hello_world"
```

You can also use `tiktoken` directly, but you would need to handle the special tokens and regex pattern yourself:

```python
import tiktoken

vocab = bpeasy.train_bpe(...)
special_tokens = ["<s>", "<pad>", "</s>"]

# Sort the vocab by rank
sorted_vocab = sorted(list(vocab.items()), key=lambda x: x[1])

# add special tokens
special_token_ranks = {}
for special_token in special_tokens:
    special_token_ranks[special_token] = len(sorted_vocab)
    sorted_vocab.append((special_token.encode("utf-8"), len(sorted_vocab)))

full_vocab = dict(sorted_vocab)

encoder = tiktoken.Encoding(
            name=name,
            pat_str=regex_pattern,
            mergeable_ranks=full_vocab,
            special_tokens=special_token_ranks,
        )
```

### Save/Load tokenizer from file

We provide basic utility functions to save and load the tokenizer from a json file.

```python
tokenizer.save("path_to_file.json")

tokenizer = BPEasyTokenizer.from_file("path_to_file.json")
```

### Export to HuggingFace format

We also support exporting the tokenizer to the HuggingFace format, which can then be used directly with the HuggingFace `transformers` library.

```python
from bpeasy.tokenizer import BPEasyTokenizer
from trans
tokenizer = BPEasyTokenizer(
    ...
)

tokenizer.export_to_huggingface_format("hf_tokenizer.json")

from transformers import PreTrainedTokenizerFast

hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file="hf_tokenizer.json")
```

### Export vocab to `tiktoken` txt format

```python
from bpeasy import 
vocab = bpeasy.train_bpe(...)

# saves the vocab to a tiktoken txt file format
save_vocab_to_tiktoken(vocab, "vocab.txt", special_tokens=["<s>", "<pad>", "</s>"])

```

If you want to use the `tiktoken` txt format, you will still need to handle the regex and special tokens yourself, as shown above,

## Contributing

Contributions are welcome! Please open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License.
