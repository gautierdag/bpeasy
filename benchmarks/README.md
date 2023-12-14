# Benchmarks on the c4 dataset

Using varying vocab sizes from (5k:100k)

| Library/Operation          | Time (seconds)                  | Standard Deviation             |
|----------------------------|---------------------------------|--------------------------------|
| HuggingFace Train          | 0.7369926854183799              | ±1.5505802971183824            |
| BPEasy Train               | 0.652837401942203               | ±0.3869646389606906            |
| HuggingFace Encode         | 0.6247405001991674              | ±0.05148973336182687           |
| BPEasy Encode (uses `tiktoken`)              | 0.26793742179870605             | ±0.03566062026595773           |

|           | Normalised Bytes/Token                  | Standard Deviation             |
|----------------------------|---------------------------------|--------------------------------|
| BPEasy Bytes/Token vs HF   | 1.0008992687171223              | ±5.542696043278318e-05         |

## Reproducing the benchmarks

```bash
pip install tokenizers
pip install bpeasy

python benchmarks/benchmark.py
```
