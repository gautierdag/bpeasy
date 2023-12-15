import base64
import json
from unittest import mock
from bpeasy.tokenizer import BPEasyTokenizer


def test_initialization():
    vocab = {b"hello": 1, b"world": 2}
    tokenizer = BPEasyTokenizer(vocab=vocab)
    assert tokenizer.vocab == vocab
    assert tokenizer.name == "bpeasy"
    assert len(tokenizer.special_tokens) == 0
    assert len(tokenizer) == 2


def test_encode_decode():
    vocab = {b"hello": 1, b" world": 2}
    tokenizer = BPEasyTokenizer(vocab=vocab)
    encoded = tokenizer.encode("hello world", allowed_special="all")
    assert encoded == [1, 2]
    decoded = tokenizer.decode(encoded)
    assert decoded == "hello world"


def test_save_and_load():
    vocab = {b"hello": 1, b" world": 2}
    tokenizer = BPEasyTokenizer(vocab=vocab)

    # Test saving
    with mock.patch("builtins.open", mock.mock_open()) as mock_file:
        tokenizer.save("dummy_path.json")
        mock_file.assert_called_once_with("dummy_path.json", "w")

    # Prepare dummy file content for loading
    dummy_file_content = json.dumps(
        {
            "name": "bpeasy",
            "vocab": {
                base64.b64encode(key).decode("utf-8"): value
                for key, value in vocab.items()
            },
            "regex_pattern": tokenizer.regex_pattern,
            "special_tokens": tokenizer.special_tokens,
        }
    )

    # Test loading
    with mock.patch(
        "builtins.open", mock.mock_open(read_data=dummy_file_content)
    ) as mock_file:
        loaded_tokenizer = BPEasyTokenizer.from_file("dummy_path.json")
        assert loaded_tokenizer.vocab == vocab


@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch("json.dump")
def test_conversion_to_huggingface(mock_json_dump, mock_open):
    vocab = {
        b"h": 0,
        b"e": 1,
        b"l": 2,
        b"o": 3,
        b" ": 4,
        b"w": 5,
        b"r": 6,
        b"d": 7,
        b"he": 8,
        b"ll": 9,
        b"llo": 10,
        b"hello": 11,
        b"wo": 12,
        b"wor": 13,
        b"ld": 14,
        b"world": 15,
        b" world": 16,
    }
    tokenizer = BPEasyTokenizer(vocab=vocab)
    tokenizer.export_to_huggingface_format("dummy_path.json")
    mock_open.assert_called_once_with("dummy_path.json", "w", encoding="utf-8")
    mock_json_dump.assert_called_once()
    args, _ = mock_json_dump.call_args
    assert args[0]["model"]["type"] == "BPE"
