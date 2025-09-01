import json
from collections.abc import Iterator, Iterable
import regex as re
import datetime
import os
from functools import lru_cache
import multiprocessing as mp



# To test your Tokenizer against our provided tests, you will first need to implement the test adapter
# at [adapters.get_tokenizer]. Then, run uv run pytest tests/test_tokenizer.py.
class Tokenizer:

    def __init__(self, vocab: dict, id_of: dict, merges: list[tuple], special_tokens:list[str]=None, num_processes: int = 10, mini_chunk_size: int = 4096,
                 desired_num_chunks: int = 1):
        # Construct a tokenizer from a given
        # vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept the following parameters:
        # vocab: dict[int, bytes]
        # merges: list[tuple[bytes, bytes]]
        # special_tokens: list[str] | None = None
        self.vocab = vocab
        self.id_of = id_of
        self.merges = merges
        self.special_tokens = list(set(['<|endoftext|>'] + (special_tokens or [])))
        self.split_special = re.compile('('+'|'.join([re.escape(token) for token in self.special_tokens])+')')
        self.num_processes = num_processes
        self.desired_num_chunks = desired_num_chunks
        self.mini_chunk_size = mini_chunk_size
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                              flags=re.UNICODE)
        self.SINGLE = [bytes([i]) for i in range(256)]
        self.rank: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        # (in the same format that your BPE training code output) and (optionally) a list of special
        # tokens. This method should accept the following additional parameters:
        # vocab_filepath: str
        # merges_filepath: str
        # special_tokens: list[str] | None = None
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
            vocab = {int(k): v.encode("latin1") for k, v in vocab_json.items()}
            id_of: dict[bytes, int] = {b: i for i, b in vocab.items()}  # bytes -> id

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_json = json.load(f)
            merges = [(l.encode("latin1"), r.encode("latin1")) for (l, r) in merges_json]

        return cls(vocab=vocab, id_of=id_of, merges=merges, special_tokens=special_tokens)


    def _split_chunks_file(self, input_path) -> list[int]:
        start_time = datetime.datetime.now()
        split_special_token = self.special_tokens[0].encode('utf-8')
        print('START SPLITTING INPUT FILE INTO CHUNKS')
        print(f'Input file: {input_path}')
        print(f'Desired number of chunks: {self.desired_num_chunks}')
        print(f'Mini chunk size: {self.mini_chunk_size}')
        print(f'Special token: {split_special_token}')
        assert isinstance(split_special_token, bytes), 'Must represent special token as a bytestring'
        with open(input_path, 'rb') as file:

            # Get total file size in bytes
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            chunk_size = file_size // self.desired_num_chunks

            # Initial guesses for chunk boundary locations, uniformly spaced
            # Chunks start on previous index, don't include last index
            chunk_boundaries = [i * chunk_size for i in range(self.desired_num_chunks + 1)]
            chunk_boundaries[-1] = file_size

            for bi in range(1, len(chunk_boundaries) - 1):
                initial_position = chunk_boundaries[bi]
                file.seek(initial_position)  # Start at boundary guess
                while True:
                    mini_chunk = file.read(self.mini_chunk_size)  # Read a mini chunk

                    # If EOF, this boundary should be at the end of the file
                    if mini_chunk == b"":
                        chunk_boundaries[bi] = file_size
                        break

                    # Find the special token in the mini chunk
                    found_at = mini_chunk.find(split_special_token)
                    if found_at != -1:
                        chunk_boundaries[bi] = initial_position + found_at + len(split_special_token)
                        break
                    initial_position += self.mini_chunk_size

            end_time = datetime.datetime.now()
            print(f'Time to split input file into chunks: {end_time - start_time}')
            # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
            return sorted(set(chunk_boundaries))

    def _split_chunks_text(self, text) -> list[int]:
        start_time = datetime.datetime.now()
        text_len = len(text)
        split_special_token = self.special_tokens[0]
        print('START SPLITTING INPUT TEXT INTO CHUNKS')
        print(f'Input text length: {text_len}')
        print(f'Desired number of chunks: {self.desired_num_chunks}')
        print(f'Mini chunk size: {self.mini_chunk_size}')
        print(f'Special token: {split_special_token}')
        assert isinstance(split_special_token, str), 'Must represent special token as a str'


        chunk_size = text_len // self.desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(self.desired_num_chunks + 1)]
        chunk_boundaries[-1] = text_len

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            next_token_position = text[initial_position:].find(split_special_token)
            if next_token_position == -1:
                chunk_boundaries[bi] = text_len
                break
            else:
                chunk_boundaries[bi] = next_token_position + len(split_special_token)


        end_time = datetime.datetime.now()
        print(f'Time to split input text into chunks: {end_time - start_time}')
        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


    def encode_parallel(self, input_path:str=None, text:str=None):
        if input_path:
            boundaries = self._split_chunks_file(input_path)
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_args.append((start, end))

            # TODO this should be done in parallel
            start, end = chunk_args[0]
            with open(input_path, 'rb') as file:
                file.seek(start)
                chunk_txt = file.read(end - start).decode("utf-8", errors="ignore")
            print(f'input text: {chunk_txt}')
            print('='*40)
            encoded_sequence = self.encode(chunk_txt)
            print(f'Encoded sequence length: {len(encoded_sequence)}')
            print(f'encoded sequence: {encoded_sequence}')
            print('='*40)
            test = []
            for e in encoded_sequence:
                test.append(self.vocab[e].decode('utf-8'))
            test_str = ''.join(test)
            print(test_str)
            print('='*40)
            print(chunk_txt == test_str)

        elif text:
            boundaries = self._split_chunks_text(text)
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_args.append((start, end))
            # TODO this should be done in parallel
            start = chunk_args[0][0]
            end = chunk_args[-1][1]
            encoded_sequence = self.encode(text[start:end])
            print(f'Encoded sequence length: {len(encoded_sequence)}')
            print(f'encoded sequence: {encoded_sequence}')

    def encode(self, text: str) -> list[int]:
        # Encode an input text into a sequence of token IDs.
        out: list[int] = []

        # Get all words' parts using regex. keep special tokens as they are
        text_by_special_tokens = self.split_special.split(text)

        for text_chunk in text_by_special_tokens:
            if not text_chunk:
                continue
            if text_chunk in self.special_tokens:
                out.append(self.id_of[text_chunk.encode("utf-8")])
                continue
            for word in self.PAT.finditer(text_chunk):
                word_b = word.group().encode('utf-8')
                out.extend(self._encode_pretoken_bytes_cached(word_b))

        return out

    @lru_cache(maxsize=100_000)
    def _encode_pretoken_bytes_cached(self, word_b: bytes) -> tuple[int, ...]:
        return tuple(self._encode_pretoken_bytes_greedy(word_b))

    def _encode_pretoken_bytes_greedy(self, word_b: bytes) -> list[int]:
        n = len(word_b)
        if n == 0:
            return []
        if n == 1:
            return [self.id_of[self.SINGLE[word_b[0]]]]

        syms = [self.SINGLE[x] for x in word_b]  # list[bytes] of length-1 chunks
        # repeatedly merge lowest-rank adjacent pair
        while True:
            best_i = -1
            best_rank = None
            for i in range(len(syms) - 1):
                r = self.rank.get((syms[i], syms[i + 1]))
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_i = i
            if best_i < 0:
                break
            syms[best_i:best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        return [self.id_of[s] for s in syms]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of
        # strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        # required for memory-eﬀicient tokenization of large files that we cannot directly load into
        # memory.
        for text in iterable:
            text_by_special_tokens = self.split_special.split(text)

            for text_chunk in text_by_special_tokens:
                if not text_chunk:
                    continue
                if text_chunk in self.special_tokens:
                    yield self.id_of[text_chunk.encode("utf-8")]
                    continue
                for word in self.PAT.finditer(text_chunk):
                    word_b = word.group().encode('utf-8')
                    yield from self._encode_pretoken_bytes_cached(word_b)

    def decode(self, ids: list[int]) -> str:
        # Decode a sequence of token IDs into text.
        decoded_chars = []
        for id in ids:
            decoded_chars.append(self.vocab[id].decode('utf-8'))
        return ''.join(decoded_chars)


test = Tokenizer.from_files(r'/Users/maksymlytvynenko/Work/Stanford/CS336/Assignment1-basics/cs336_basics/vocab_ts.json',
                                r'/Users/maksymlytvynenko/Work/Stanford/CS336/Assignment1-basics/cs336_basics/merges_ts.json',
                                )
test.encode_parallel(input_path=r'/Users/maksymlytvynenko/Work/Stanford/CS336/Assignment1-basics/data/debug_couple_stories.txt')
text_test = '''Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed!
He said, “Wow, that is a really amazing vase! Can I buy it?”
The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!”
So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was.
And that's how Ben found an amazing vase in the store!
<|endoftext|>
Once upon a time, there was a reliable otter named Ollie. He lived in a river with his family. They all loved to play and swim together.
One day, Ollie's mom said, "Ollie, hurry and get some fish for dinner!" Ollie swam fast to catch fish. He saw his friend, the duck. "Hi, Ollie!" said the duck. "Hi, duck!" said Ollie. "I need to hurry and catch fish for my family."
While Ollie was catching fish, he found a big shiny stone. He thought, "This is not a fish, but it is so pretty!" Ollie took the shiny stone home to show his family. They all looked at the shiny stone and smiled. The shiny stone made everyone happy, and they forgot about the fish for dinner.
<|endoftext|>'''
# test.encode_parallel(text=text_test)
