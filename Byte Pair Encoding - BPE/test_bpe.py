
import pytest

from bpe import BPETokenizer


def test_tokenizer_training():

    corpus = [
        "low lower lowest",
        "low lower",
        "low lowest"
    ]

    tokenizer = BPETokenizer(num_merges=10)

    tokenizer.train(corpus)

    assert tokenizer.is_trained is True
    assert len(tokenizer.merges) > 0
    assert len(tokenizer.vocab) > 0


def test_encode():

    corpus = [
        "low lower lowest",
        "low lower",
        "low lowest"
    ]

    tokenizer = BPETokenizer(num_merges=10)

    tokenizer.train(corpus)

    tokens = tokenizer.encode("lower")

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert "".join(tokens) == "lower"


def test_decode():

    tokenizer = BPETokenizer(num_merges=10)

    tokenizer.train(["hello world hello"])

    tokens = tokenizer.encode("hello")

    decoded = tokenizer.decode(tokens)

    assert decoded == "hello"


def test_untrained_tokenizer():

    tokenizer = BPETokenizer()

    with pytest.raises(RuntimeError):

        tokenizer.encode("hello")


def test_empty_corpus():

    tokenizer = BPETokenizer()

    tokenizer.train([])

    assert tokenizer.is_trained is True
    assert tokenizer.merges == []


def test_get_vocab():

    tokenizer = BPETokenizer(num_merges=5)

    tokenizer.train(["hello hello world"])

    vocab = tokenizer.get_vocab()

    assert isinstance(vocab, dict)
    assert len(vocab) > 0


def test_unknown_characters():

    tokenizer = BPETokenizer(num_merges=5)

    tokenizer.train(["hello world"])

    tokens = tokenizer.encode("xyz")

    assert "".join(tokens) == "xyz"

def test_whitespace_preservation():

    corpus = [
        "hello world",
        "hello   world",
        "hello\nworld"
    ]

    tokenizer = BPETokenizer(num_merges=10)

    tokenizer.train(corpus)

    text = "hello   world\nhello"

    tokens = tokenizer.encode(text)

    decoded = tokenizer.decode(tokens)

    assert decoded == text


def test_frequency_weighted_vocabulary():

    corpus = [
        "hello hello hello",
        "world"
    ]

    tokenizer = BPETokenizer(num_merges=0)

    tokenizer.train(corpus)

    vocab = tokenizer.get_vocab()

    # Character 'h' occurs three times in hello
    assert vocab["h"] == 3

    # Character 'w' occurs once in world
    assert vocab["w"] == 1
