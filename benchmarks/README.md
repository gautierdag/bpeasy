# Benchmarks on the c4 dataset

Using varying vocab sizes from (5k:100k)

| Library/Operation          | Time (seconds)                  | Standard Deviation             |
|----------------------------|---------------------------------|--------------------------------|
| HuggingFace Train          | 0.7369              | ±1.55            |
| `bpeasy` Train               | 0.6528               | ±0.386            |
| HuggingFace Encode         | 0.6247              | ±0.051           |
| `bpeasy` Encode (uses `tiktoken`)              | 0.2679             | ±0.035           |

|           | Bytes per Token (normalised against HF)                  | Standard Deviation             |
|----------------------------|---------------------------------|--------------------------------|
| `bpeasy`   | 1.0008992687171223              | ±5.542696043278318e-05         |

We can see that BPEasy is faster than HuggingFace for training and encoding. Though the difference is not massive for training and will heavily depend on the dataset/compute, it is comparable.

We also gain a tiny bit of compression (more bytes per token) because `bpeasy` works at the byte level and is slightly more efficient in its allocation of basic tokens.

## Reproducing the benchmarks

```bash
pip install tokenizers
pip install bpeasy

python benchmarks/train.py
```
