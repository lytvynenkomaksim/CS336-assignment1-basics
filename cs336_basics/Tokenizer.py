from collections.abc import Iterator, Iterable


# To test your Tokenizer against our provided tests, you will first need to implement the test adapter
# at [adapters.get_tokenizer]. Then, run uv run pytest tests/test_tokenizer.py.
class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        # Construct a tokenizer from a given
        # vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept the following parameters:
        # vocab: dict[int, bytes]
        # merges: list[tuple[bytes, bytes]]
        # special_tokens: list[str] | None = None
        pass

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Class
        # method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        # (in the same format that your BPE training code output) and (optionally) a list of special
        # tokens. This method should accept the following additional parameters:
        # vocab_filepath: str
        # merges_filepath: str
        # special_tokens: list[str] | None = None
        pass

    def encode(self, text: str) -> list[int]:
        # Encode an input text into a sequence of token IDs.
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of
        # strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        # required for memory-eﬀicient tokenization of large files that we cannot directly load into
        # memory.
        pass

    def decode(self, ids: list[int]) -> str:
        # Decode a sequence of token IDs into text.
        pass