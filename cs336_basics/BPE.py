import os
import multiprocessing as mp
from collections import Counter
import datetime
import regex as re


# def bytes_to_unicode():
#     """
#     Returns a dictionary mapping bytes (0-255) to unicode strings.
#     This is the mapping used by GPT-2. It shifts unprintable bytes to a different
#     unicode range to make them all printable.
#     """
#     # The first ranges are the printable ASCII characters
#     bs = (
#             list(range(ord("!"), ord("~") + 1)) +
#             list(range(ord("¡"), ord("¬") + 1)) +
#             list(range(ord("®"), ord("ÿ") + 1))
#     )
#
#     cs = bs[:]
#     n = 0
#     # The loop handles the unprintable bytes
#     for b in range(2 ** 8):
#         if b not in bs:
#             bs.append(b)
#             cs.append(2 ** 8 + n)
#             n += 1
#
#     cs = [chr(n) for n in cs]
#
#     return dict(zip(bs, cs))

class BPE:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str] = None, num_processes: int = 10, mini_chunk_size: int = 4096,
                 desired_num_chunks: int = 40):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = list(set(['<|endoftext|>'] + (special_tokens or [])))
        self.split_special = re.compile('|'.join([re.escape(token) for token in self.special_tokens]))
        self.vocab = {}
        self.merges = []
        self.num_processes = num_processes
        self.desired_num_chunks = desired_num_chunks
        self.mini_chunk_size = mini_chunk_size
        self.data_chunks_boundaries = self._split_chunks()
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", flags=re.UNICODE)

        self._init_fill_vocab()


    def _init_fill_vocab(self):
        # This works for 2 out of 3 tests
        for i in range(256):
            self.vocab[i] = bytes([i])

        next_id = len(self.vocab)

        for token_str in self.special_tokens:
            self.vocab[next_id] = token_str.encode("utf-8")
            next_id += 1


    # def _init_fill_vocab(self):
    #     """
    #     Initializes the vocabulary using the specific GPT-2 byte mapping,
    #     and places special tokens at the beginning.
    #     """
    #     # 1. Add special tokens first, starting from ID 0.
    #     for i, token_str in enumerate(self.special_tokens):
    #         self.vocab[i] = token_str.encode("utf-8")
    #
    #     # 2. Create the GPT-2 byte-to-unicode mapping.
    #     byte_encoder = bytes_to_unicode()
    #     unicode_to_byte = {v: k for k, v in byte_encoder.items()}
    #
    #     # 3. Add the 256 base byte tokens, starting after the special tokens.
    #     #    The token ID is its rank in the sorted list of unicode characters.
    #     start_id = len(self.vocab)
    #     for i in range(256):
    #         # Get the unicode character for this byte from the mapping
    #         unicode_char = byte_encoder[i]
    #         # The token ID is its rank + the starting ID
    #         token_id = list(sorted(unicode_to_byte)).index(unicode_char) + start_id
    #         self.vocab[token_id] = bytes([i])


    def get_vocab(self) -> dict[int, bytes]:
        return self.vocab

    def _split_chunks(self) -> list[int]:
        start_time = datetime.datetime.now()
        split_special_token = self.special_tokens[0].encode('utf-8')
        print('START SPLITTING INPUT FILE INTO CHUNKS')
        print(f'Input file: {self.input_path}')
        print(f'Desired number of chunks: {self.desired_num_chunks}')
        print(f'Mini chunk size: {self.mini_chunk_size}')
        print(f'Special token: {split_special_token}')
        assert isinstance(split_special_token, bytes), 'Must represent special token as a bytestring'
        with open(self.input_path, 'rb') as file:

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
                        chunk_boundaries[bi] = initial_position + found_at
                        break
                    initial_position += self.mini_chunk_size

            end_time = datetime.datetime.now()
            print(f'Time to split input file into chunks: {end_time - start_time}')
            # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
            return sorted(set(chunk_boundaries))


    def _process_chunk_and_get_pairs(self, args):
        start, end = args

        # --- Part 1: Get word counts (from your _process_chunk) ---
        start_time = datetime.datetime.now()
        word_counts = Counter()
        with open(self.input_path, 'rb') as file:
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            chunks_by_special_tokens = self.split_special.split(chunk)
            for special_chunk in chunks_by_special_tokens:
                words_generator = (m.group(0) for m in self.PAT.finditer(special_chunk))
                word_counts.update(words_generator)
        end_time = datetime.datetime.now()
        # print(f'Time to process chunk and get word counts: {end_time - start_time}')

        # --- Part 2: Get pair counts (from your _get_pairs_from_chunk) ---
        word_structures_chunk = {}
        start_time = datetime.datetime.now()
        pair_counts = Counter()
        for word_str, count in word_counts.items():
            word_bytes_tuple = tuple(word_str.encode('utf-8'))
            word_structures_chunk[word_bytes_tuple] = count

            if len(word_bytes_tuple) < 2:
                continue
            for i in range(len(word_bytes_tuple) - 1):
                pair = (word_bytes_tuple[i], word_bytes_tuple[i + 1])
                pair_counts[pair] += count
        end_time = datetime.datetime.now()
        # print(f'Time to get pairs from chunk: {end_time - start_time}')

        return (word_structures_chunk, pair_counts)


    def train_merges(self, word_structures: dict, final_pair_counts: dict):
        num_merges_needed = self.vocab_size - len(self.vocab)
        if num_merges_needed <= 0:
            print('Vocabulary size already met or exceeded. No merges needed.')
            return 0

        print(f'\n--- Starting BPE Merges (need {num_merges_needed}) ---')

        for i in range(num_merges_needed):
            # print(f'Starting merge {i + 1}/{num_merges_needed}...')
            if not final_pair_counts:
                print('No more pairs to merge. Stopping early.')
                break

            # best_pair = max(final_pair_counts, key=lambda pair: (final_pair_counts[pair], pair))
            best_pair = max(final_pair_counts,
                            key=lambda p: (final_pair_counts[p], self.vocab[p[0]], self.vocab[p[1]])
                        )
            # print(f'Best pair to merge: {best_pair} with count {final_pair_counts[best_pair]}')
            # print(f' Best pair bytes: {self.vocab[best_pair[0]]} + {self.vocab[best_pair[1]]}')
            # print(f' my token1_bytes: {self.vocab[best_pair[0]]}, token2_bytes: {self.vocab[best_pair[1]]}')
            token1_bytes = self.vocab[best_pair[0]]
            token2_bytes = self.vocab[best_pair[1]]
            new_token_bytes = token1_bytes + token2_bytes
            if new_token_bytes == b'ng':
                print('Found "ng" token.')
                print('merge number :', i + 1)
                print('Best pair:', best_pair)
                print('final_pair_counts:', final_pair_counts.most_common(5))
                print('len(self.vocab):', len(self.vocab))
            # print(f' New token bytes: {new_token_bytes}')
            # print('=' * 20)


            new_token_id = len(self.vocab)
            self.vocab[new_token_id] = new_token_bytes
            self.merges.append((token1_bytes, token2_bytes))
            # if len(self.merges) == 93:
            #     print('='*20)
            #     print(final_pair_counts.most_common(5))
            #     print('Best pair:', best_pair)
            #     print('=' * 20)

            word_structures, final_pair_counts = self._merge_and_update_counts(
                word_structures,
                final_pair_counts,
                best_pair,
                new_token_id
            )
            # word_structures = self._merge_and_update_counts(
            #     word_structures,
            #     final_pair_counts,
            #     best_pair,
            #     new_token_id
            # )
            # print('New word structures after merge:')
            # for word in word_structures:
            #     print(f'Word: {word}, Count: {word_structures[word]}')

            # if (i + 1) % 50 == 0:
            #     print(f'  Merge {i + 1}/{num_merges_needed}: Merged {best_pair} -> {new_token_id} ({new_token_bytes})')

        print(f'--- BPE training complete. Final vocabulary size: {len(self.vocab)} ---')



    def _merge_and_update_counts(self, word_structures: dict, pair_counts: dict, best_pair: tuple,
                                 new_token_id: int) -> tuple[dict, dict]:
        byte_1, byte_2 = best_pair

        # Accumulate pair count changes
        pairs_to_decrement = Counter()
        pairs_to_increment = Counter()
        words_to_increment = Counter()
        words_to_decrement = Counter()
        # print(f'Merging pair: {byte_1} + {byte_2} into new token ID: {new_token_id}')
        temp_dict_to_replace = {}
        for word, count in word_structures.items():
            # print(f'word: {word}, count: {count}')
            if len(word) < 2:
                continue
            if byte_1 in word and byte_2 in word:
                new_word, old_pairs, new_pairs = self._merge_word_with_index(word, best_pair, new_token_id)
                if new_word != word:  # Only update if the word actually changed
                    words_to_increment[new_word] += count
                    words_to_decrement[word] += count

                    for old_pair in old_pairs:
                        pairs_to_decrement[old_pair] += count
                    for new_pair in new_pairs:
                        pairs_to_increment[new_pair] += count

        for word in words_to_increment:
            word_structures[word] = word_structures.get(word, 0) + words_to_increment[word]
        for word in words_to_decrement:
            del word_structures[word]
        for pair in pairs_to_increment:
            pair_counts[pair] += pairs_to_increment[pair]
        for pair in pairs_to_decrement:
            pair_counts[pair] -= pairs_to_decrement[pair]

        del pair_counts[best_pair] # we merged this pair in all words, so remove it from counts

        return word_structures, pair_counts

    def _merge_word_with_index(self, word: tuple, best_pair: tuple, new_token_id: int) -> tuple[tuple, set, set]:
        byte_1, byte_2 = best_pair

        new_word = []
        old_pairs = []
        new_pairs = []

        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                # Found the pair to merge!

                # If there's a token before, the pair (prev, byte_1) becomes (prev, new_token_id)
                if i > 0:
                    old_pairs.append((word[i - 1], byte_1))
                    new_pairs.append((word[i - 1], new_token_id))

                # If there's a token after, the pair (byte_2, next) becomes (new_token_id, next)
                if i + 2 < len(word):
                    old_pairs.append((byte_2, word[i + 2]))
                    new_pairs.append((new_token_id, word[i + 2]))

                # Add the merged token
                new_word.append(new_token_id)
                i += 2  # Skip both bytes of the merged pair
            else:
                new_word.append(word[i])
                i += 1

        return tuple(new_word), set(old_pairs), set(new_pairs)



    # def _merge_and_update_counts(self, word_structures, pair_counts, best_pair, new_token_id)-> dict:
    #     new_word_structures = {}
    #     affected_words = {}  # Store {old_word: new_word} for words that changed
    #
    #     # --- Pass 1: Find all affected words and create their new structure ---
    #     for word, count in word_structures.items():
    #         if len(word) < 2:
    #             new_word_structures[word] = count
    #             continue
    #
    #         # Build the new word by replacing all instances of the best_pair
    #         i = 0
    #         new_word = []
    #         has_changed = False
    #         while i < len(word):
    #             if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
    #                 new_word.append(new_token_id)
    #                 i += 2
    #                 has_changed = True
    #             else:
    #                 new_word.append(word[i])
    #                 i += 1
    #
    #         if has_changed:
    #             # This word was affected. Store the old and new versions.
    #             affected_words[word] = tuple(new_word)
    #         else:
    #             # This word was not affected, so it carries over directly.
    #             new_word_structures[word] = count
    #
    #     # --- Pass 2: Update counts and the final structure based on affected words ---
    #     for old_word, new_word in affected_words.items():
    #         count = word_structures[old_word]
    #
    #         # Decrement counts for all pairs in the old word
    #         for i in range(len(old_word) - 1):
    #             pair = (old_word[i], old_word[i + 1])
    #             pair_counts[pair] -= count
    #             if pair_counts[pair] == 0:
    #                 del pair_counts[pair]
    #
    #         # Increment counts for all pairs in the new word
    #         for i in range(len(new_word) - 1):
    #             pair = (new_word[i], new_word[i + 1])
    #             pair_counts[pair] = pair_counts.get(pair, 0) + count
    #
    #         # Add the new, merged word to the final structures, aggregating counts
    #         new_word_structures[new_word] = new_word_structures.get(new_word, 0) + count
    #
    #     # The original pair has now been fully replaced and its count should be zero.
    #     if best_pair in pair_counts:
    #         del pair_counts[best_pair]
    #
    #     return new_word_structures

    def run_BPE(self):
        print(f'Number of actual chunks {len(self.data_chunks_boundaries)-1} chunks.')
        print(f'Chunk boundaries: {self.data_chunks_boundaries}')

        available_cores = mp.cpu_count()
        print(f'Available cores: {available_cores}')
        print(f'Desired number of processes: {self.num_processes}')

        chunk_args = []
        for start, end in zip(self.data_chunks_boundaries[:-1], self.data_chunks_boundaries[1:]):
            chunk_args.append((start, end))

        print('Starting combined processing pool...')
        start_time = datetime.datetime.now()
        with mp.Pool(self.num_processes) as pool:
            list_of_results = pool.map(self._process_chunk_and_get_pairs, chunk_args)
        end_time = datetime.datetime.now()
        print(f'Time to process all chunks and get pairs: {end_time - start_time}')

        print('Combining results...')
        start_time = datetime.datetime.now()

        # These will be the complete data structures you need for the merge loop
        word_structures = {}
        final_pair_counts = Counter()

        for word_struct_chunk, pair_count_chunk in list_of_results:
            for word, count in word_struct_chunk.items():
                word_structures[word] = word_structures.get(word, 0) + count

            final_pair_counts.update(pair_count_chunk)

        end_time = datetime.datetime.now()
        print(f'Time to combine all structures: {end_time - start_time}')

        # print(f'Initial unique pairs: {len(final_pair_counts)}')

        # print('='*20)
        # print('Initial word structures:', word_structures)
        print('\n')
        # print(f'Top 15 initial pairs: {final_pair_counts.most_common(15)}')

        self.train_merges(word_structures, final_pair_counts)


def train_BPE_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[
    dict[int, bytes], list[tuple[bytes, bytes]]]:
    # You can pass other parameters like num_processes if you want
    tokenizer = BPE(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)

    # Run the main training logic
    tokenizer.run_BPE()  # This will call train_merges internally
    print(tokenizer.get_vocab())
    print('=' * 20)
    print(tokenizer.merges)
    # Return the final vocabulary and merges in the required format
    return tokenizer.get_vocab(), tokenizer.merges


if __name__ == '__main__':
    start = datetime.datetime.now()
    # test = BPE(input_path='../data/debug_small_text.txt', vocab_size=270)
    # test = BPE(input_path='../data/smallest.txt', vocab_size=270)
    # test = BPE(input_path='../data/owt_train.txt', vocab_size=300)
    test = BPE(input_path='/Users/maksymlytvynenko/Work/Stanford/CS336/Assignment1-basics/tests/fixtures/corpus.en',
               vocab_size=500,
               special_tokens=["<|endoftext|>"])
    test.run_BPE()
    end = datetime.datetime.now()
    print(f'Total time: {end - start}')
    print(test.get_vocab())
    print('='*20)
    print(test.merges)



