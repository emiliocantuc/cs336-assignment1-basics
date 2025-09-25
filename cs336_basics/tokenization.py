import os
import regex as re
from collections import Counter
from typing import BinaryIO
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
            pretokenized[tuple(b[j:j+1] for j in range(len(b)))] += 1

    return pretokenized

def pretokenize(input_path: str | os.PathLike, special_tokens: list[str], chunk_size: int, end_of_doc_token: str) -> dict[tuple[bytes], int]:

    # Pretokenize by scanning the whole file
    pretokenized: dict[tuple[bytes], int] = Counter()
    special_pat = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))


    file_size = os.path.getsize(input_path)
    N_CHUNKS = (file_size + chunk_size - 1) // chunk_size

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
    
    def get_pair_freq(): # Function for profiling
        # Build pair frequencies
        for bs, freq in pretokenized.items():
            for i in range(0, len(bs) - 1):
                k = bs[i : i + 2]
                pair_freq[k] += freq
    get_pair_freq()

    def get_most_freq():
        return max(
            pair_freq, key=lambda x: (pair_freq[x], *x)
        )  # clever way to break ties: second criterion as second item in tuple

    while len(vocab) < vocab_size:

        most_freq = get_most_freq()
        del pair_freq[most_freq]

        a, b = most_freq

        merges.append(most_freq)
        vocab.append(a + b)

        # in the pretokenized dict, replace keys (..., a, b, ...) with (..., ab, ...)
        for bs in list(pretokenized.keys()):
            pre_freq = pretokenized.get(bs, 0)

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
                        pair_freq[bs[i - 1], a + b] += pre_freq

                    if i < len(bs) - 2:
                        pair_freq[a + b, bs[i + 2]] += pre_freq

                    # remove (prev, a), (b, next)
                    if i > 0:
                        k = (bs[i - 1], a)
                        pair_freq[k] -= pre_freq
                        if pair_freq[k] <= 0:
                            del pair_freq[k]

                    if i < len(bs) - 2:
                        k = (b, bs[i + 2])
                        pair_freq[k] -= pre_freq
                        if pair_freq[k] <= 0:
                            del pair_freq[k]

                    bs = new_bs
                else:
                    i += 1

    return {i: b for i, b in enumerate(vocab)}, merges


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
    import json
    from functools import partial

    input_path = "tests/fixtures/tinystories_sample_5M.txt"
    vocab_size = 1_000

    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    # input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10_000

    d = partial(bytes.decode, encoding="utf-8", errors="ignore")

    os.makedirs("results", exist_ok=True)

    def run():
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
            chunk_size=8192 * 4,  # 32kB chunks
        )
        with open("results/bpe_vocab.json", "w") as f:
            json.dump({i: d(v) for i, v in vocab.items()}, f, indent=2)

        with open("results/bpe_merges.json", "w") as f:
            json.dump([(d(a), d(b)) for a,b in merges], f, indent=2)

    cProfile.run("run()")