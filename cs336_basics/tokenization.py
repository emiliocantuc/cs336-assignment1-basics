import os
import regex as re
from collections import Counter
import heapq
from typing import BinaryIO, Iterable
from multiprocessing import Pool, cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_chunk(input_path: str | os.PathLike, start: int, end: int, special_pat: str) -> dict[tuple[bytes], int]:
    pretokenized: dict[tuple[bytes], int] = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    parts = re.split(special_pat, chunk) if special_pat else [chunk]
    for part in parts:
        for m in re.finditer(PAT, part):
            s = m.group(0)
            b = s.encode("utf-8")
            pretokenized[tuple(b[j : j + 1] for j in range(len(b)))] += 1

    return pretokenized


def pretokenize(
    input_path: str | os.PathLike, special_tokens: list[str], chunk_size: int, end_of_doc_token: str
) -> dict[tuple[bytes], int]:

    pretokenized: dict[tuple[bytes], int] = Counter()
    special_pat = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))

    file_size = os.path.getsize(input_path)
    N_CHUNKS = (file_size + chunk_size - 1) // chunk_size
    # print(f"File size: {file_size} bytes, chunk size: {chunk_size} bytes, N_CHUNKS: {N_CHUNKS}")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, N_CHUNKS, end_of_doc_token.encode())

        with Pool(min(cpu_count(), N_CHUNKS)) as p:
            results = [
                p.apply_async(pretokenize_chunk, (input_path, start, end, special_pat))
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
            for r in results:
                pretokenized.update(r.get())

    return pretokenized


# TODO have to use this?
class Desc:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

    def __lt__(self, other):
        return self.x > other.x

    def __repr__(self):
        return f"Desc({self.x!r})"


class MostFrequentPairHeap:
    def __init__(self, pair_freq: Counter):
        self.pair_freq = pair_freq
        self.freq_heap = [Desc((freq, a, b)) for (a, b), freq in pair_freq.items()]
        heapq.heapify(self.freq_heap)

    def pop(self):
        while self.freq_heap:
            freq, a, b = heapq.heappop(self.freq_heap).x
            # freq = -neg_freq
            if freq == self.pair_freq.get((a, b)) and freq > 0:
                self.pair_freq.pop((a, b), None)
                return a, b
        return None

    def update(self, pair, freq_delta):
        if freq_delta == 0:
            return

        freq = self.pair_freq.get(pair, 0) + freq_delta
        if freq > 0:
            self.pair_freq[pair] = freq
            a, b = pair
            heapq.heappush(self.freq_heap, Desc((freq, a, b)))
        else:
            self.pair_freq.pop(pair, None)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    end_of_doc_token: str = "<|endoftext|>",
    chunk_size: int = 8192,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
        end_of_doc_token (str, optional): A special token that indicates the end of a document.
            This token is used to find chunk boundaries when splitting the input file into chunks.
            Defaults to "<|endoftext|>".
        chunk_size (int, optional): The size (in bytes) of each chunk to read from the input file.
            Defaults to 8192 (8kB).

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab = [bytes([i]) for i in range(256)] + [t.encode() for t in special_tokens]
    merges = []

    pretokenized = pretokenize(input_path, special_tokens, chunk_size, end_of_doc_token)

    pair_freq = Counter()

    def get_pair_freq():  # Function for profiling
        # Build pair frequencies
        for bs, freq in pretokenized.items():
            for i in range(0, len(bs) - 1):
                k = bs[i : i + 2]
                pair_freq[k] += freq

    get_pair_freq()
    most_freq_heap = MostFrequentPairHeap(pair_freq)

    while len(vocab) < vocab_size:
        most_freq = most_freq_heap.pop()

        if most_freq is None:
            print("No more pairs to merge.")
            break
        a, b = most_freq

        merges.append(most_freq)
        vocab.append(a + b)

        # in the pretokenized dict, replace keys (..., a, b, ...) with (..., ab, ...)
        for bs, pre_freq in list(pretokenized.items()):
            # pre_freq = pretokenized.get(bs, 0) # Removing this saved a lot of time

            i = 0
            while i < len(bs) - 1:
                if bs[i : i + 2] == most_freq:
                    # replace (..., a, b, ...) with (..., ab, ...)
                    new_bs = bs[:i] + (a + b,) + bs[i + 2 :]
                    pretokenized[new_bs] += pre_freq
                    del pretokenized[bs]

                    # update pair_freq

                    # add (prev, (a,b)) and ((a,b), next)
                    if i > 0:
                        most_freq_heap.update((bs[i - 1], a + b), pre_freq)

                    if i < len(bs) - 2:
                        most_freq_heap.update((a + b, bs[i + 2]), pre_freq)

                    # remove (prev, a), (b, next)
                    if i > 0:
                        k = (bs[i - 1], a)
                        most_freq_heap.update(k, -pre_freq)

                    if i < len(bs) - 2:
                        k = (b, bs[i + 2])
                        most_freq_heap.update(k, -pre_freq)

                    bs = new_bs

                else:
                    i += 1

    return {i: b for i, b in enumerate(vocab)}, merges



def bytes_to_unicode():
    # from OpenAI GPT-2 tokenization
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

_BT2U = bytes_to_unicode()
_U2BT = {u: b for b, u in _BT2U.items()}

def gpt2_str_token_to_bytes(s: str) -> bytes:
    # map each unicode char in GPT-2 token string back to its original byte
    return bytes([_U2BT[ch] for ch in s])


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens = set(self.special_tokens)
        special_pat = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
        self.special_re = re.compile("(" + special_pat + ")") if special_pat else None

        assert isinstance(self.vocab, dict)
        assert all(isinstance(k, int) and isinstance(v, bytes) for k, v in self.vocab.items())
        assert isinstance(self.merges, list)
        assert all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in self.merges)

    @classmethod
    def from_files(
        cls, vocab_filepath: str | os.PathLike, merges_filepath: str, special_tokens: list[str] | None = None
    ):
        import json

        with open(vocab_filepath) as f:
            raw = json.load(f)
        vocab = {id_: gpt2_str_token_to_bytes(tok_str) for tok_str, id_ in raw.items()}

        merges = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()
                merges.append((gpt2_str_token_to_bytes(a), gpt2_str_token_to_bytes(b)))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token ids."""

        out = []

        for part in self.special_re.splititer(text) if self.special_re else [text]:
            if not part:
                continue
            elif self.special_re and part in self.special_tokens:
                out.append(self.inv_vocab[part.encode('utf-8')])
            else:
                for m in re.finditer(PAT, part):

                    pretoken = m.group(0)
                    b = pretoken.encode('utf-8')
                    bs = tuple(b[i:i+1] for i in range(len(b)))

                    while True:

                        pairs = list((bs[i], bs[i + 1]) for i in range(len(bs) - 1))
                        if not pairs:
                            break

                        # TODO replace with a lazy min heap
                        rank, merge = min([(self.merge_rank.get(p, float('inf')), p) for p in pairs])
                        if rank == float('inf'):
                            break

                        new_bs = []
                        i = 0
                        while i < len(bs):
    
                            if i < len(bs)-1 and bs[i:i+2] == merge:
                                new_bs.append(bs[i]+bs[i+1])
                                i += 2
                            else:
                                new_bs.append(bs[i])
                                i += 1

                        bs = tuple(new_bs)

                    out.extend([self.inv_vocab[b] for b in bs])

        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""

        out = bytearray()
        for i in ids:
            out.extend(self.vocab.get(i, b'\xef\xbf\xbd'))  # U+FFFD in UTF-8
        return out.decode("utf-8", errors="replace")


# Taken verbatim from pretokenization_example.py
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


if __name__ == "__main__":
    import cProfile
    import pickle

    input_path = "tests/fixtures/tinystories_sample_5M.txt"
    vocab_size = 1_000

    # input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    # # input_path = "data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 10_000

    # os.makedirs("results", exist_ok=True)

    # def run():
    #     vocab, merges = train_bpe(
    #         input_path=input_path,
    #         vocab_size=vocab_size,
    #         special_tokens=["<|endoftext|>"],
    #         chunk_size=65536,  # 8192 * 8,  # 32kB chunks
    #     )
    #     with open("results/bpe_vocab.pkl", "wb") as f:
    #         pickle.dump(vocab, f)

    #     with open("results/bpe_merges.pkl", "wb") as f:
    #         pickle.dump(merges, f)

    # run()
    # cProfile.run("run()")

    tok = BPETokenizer.from_files("tests/fixtures/gpt2_vocab.json", "tests/fixtures/gpt2_merges.txt")
    print(tok.encode("Hello, how are you?"))
    print(tok.encode("Hóla"))
    print(tok.decode(tok.encode("Hóla")))
    import pdb; pdb.set_trace()
