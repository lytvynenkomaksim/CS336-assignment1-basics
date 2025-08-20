import json
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
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        # (in the same format that your BPE training code output) and (optionally) a list of special
        # tokens. This method should accept the following additional parameters:
        # vocab_filepath: str
        # merges_filepath: str
        # special_tokens: list[str] | None = None
        with open(vocab_filepath, 'rb') as vocab_file:
            vocab = json.loads(vocab_file.read())
            print(type(vocab))
            print(vocab)

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


test = Tokenizer.from_files(r'/Users/maksymlytvynenko/Work/Stanford/CS336/Assignment1-basics/cs336_basics/vocab_ts.txt',
                                r'/Users/maksymlytvynenko/Work/Stanford/CS336/Assignment1-basics/cs336_basics/merges_ts.txt',
                                )



def load_vocab_merges_json(vocab_path: str, merges_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vj = json.load(f)
        vocab = {int(k): v.encode("latin1") for k, v in vj.items()}

    with open(merges_path, "r", encoding="utf-8") as f:
        mj = json.load(f)
        merges = [(a.encode("latin1"), b.encode("latin1")) for (a, b) in mj]

    return vocab, merges