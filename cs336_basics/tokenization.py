import os
import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(input_path: str | os.PathLike) -> list[bytes]:
    re.finditer(
        "\s",
    )

    raise NotImplementedError


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
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

    # Pretokenize by scanning the whole file
    pretokenized: dict[bytes, int] = Counter()

    CHUNK = 8192  # read 8 KB at a time TODO: tune?
    with open(input_path, "rb") as f:
        while True:
            txt = f.read(CHUNK)
            if not txt:
                break

            for m in re.finditer(PAT.encode(), txt):
                pretokenized[tuple(m[0])] += 1

    # Build pair frequencies
    pair_freq: dict[tuple[bytes, bytes], int] = Counter()
    for bs, freq in pretokenized.items():
        for i in range(0, len(bs)-1):
            pair_freq[bs[i:i+2]] += freq


    print(pretokenized.most_common(10))
    print(pair_freq.most_common(10))
    print(len(pretokenized), len(pair_freq))

    merges = []

    (most_freq, highest_pair_freq), = pair_freq.most_common(1)
    a, b = most_freq
    a_t = a if isinstance(a, tuple) else (a,)
    b_t = b if isinstance(b, tuple) else (b,)

    merges.append(most_freq)

    print('most freq', most_freq, freq, a, b)

    for _ in range(10):#(vocab_size - 256 - len(special_tokens)):

        for bs in list(pretokenized.keys()):

            i = 0
            while i < len(bs)-1:
                if bs[i:i+2] == most_freq:

                    # replace (..., a, b, ...) with (..., ab, ...)
                    new_bs = bs[:i] + (a_t+b_t,) + bs[i+2:]
                    pre_freq = pretokenized.pop(bs)

                    print('new bs', bs, new_bs)

                    # update pair_freq
                    m = (*a_t, *b_t),

                    # add (prev, (a,b)) and ((a,b), next)
                    if i > 0:
                        prev = bs[i-1:i]
                        pair_freq[prev + m] += pre_freq

                    if i < len(bs)-2:
                        next = bs[i+2:i+3]
                        pair_freq[m + next] += pre_freq

                    # remove (prev, a), (b, next)
                    if i > 0:
                        prev = bs[i-1:i]
                        k = prev + (a,)
                        pair_freq[k] -= pre_freq
                        if pair_freq[k] <= 0:
                            del pair_freq[k]
                    
                    if i < len(bs)-2:
                        next = bs[i+2:i+3]
                        k = (b,) + next
                        pair_freq[k] -= pre_freq
                        if pair_freq[k] <= 0:
                            del pair_freq[k]
                
                    bs = new_bs
                else:
                    i += 1


    import pdb; pdb.set_trace()

    # break


if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 50
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

    train_bpe(input_path, vocab_size, special_tokens)

    # print('vocab:')
    # for i, b in vocab.items():
    #     print(f'  {i:3}: {b}')
    # print('merges:')
    # for a, b in merges:
    #     print(f'  {bytes(a)} {bytes(b)}')
