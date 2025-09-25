import os
import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# just split on whitespace
# PAT = r"""\S+"""


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

    vocab = [bytes([i]) for i in range(256)] + [t.encode() for t in special_tokens]
    merges = []

    # Pretokenize by scanning the whole file
    pretokenized: dict[tuple[bytes], int] = Counter()
    special_set = {t.encode() for t in special_tokens}
    escaped_specials = [re.escape(t).encode() for t in special_tokens]
    escaped_specials.sort(key=len, reverse=True)
    token_pattern_bytes = PAT.encode()
    if escaped_specials:
        special_branch = b"|".join(escaped_specials)
        token_pattern_bytes = b"(?:" + special_branch + b")|" + token_pattern_bytes
    token_pattern = re.compile(token_pattern_bytes)

    CHUNK = None #8192  # read 8 KB at a time TODO: tune?
    with open(input_path, "rb") as f:
        while True:
            txt = f.read(CHUNK)
            if not txt:
                break

            for m in token_pattern.finditer(txt):
                token_bytes = m[0]
                if token_bytes in special_set:
                    pretokenized[(token_bytes,)] += 1
                    continue

                k = tuple(token_bytes[i : i + 1] for i in range(len(token_bytes)))
                pretokenized[k] += 1
    
    # print('initial pretokenized', pretokenized)
    for bs in pretokenized:
        assert r'<' not in bs

    # Build pair frequencies
    pair_freq = Counter()
    for bs, freq in pretokenized.items():
        for i in range(0, len(bs)-1):
            k = bs[i:i+2]
            pair_freq[k] += freq

    # print(len(pretokenized), len(pair_freq))
    # print('='*40)

    while len(vocab) < vocab_size:

        most_freq = max(pair_freq, key=lambda x: (pair_freq.get(x, 0), *x)) # clever way to break ties: second criterion as second item in tuple
        del pair_freq[most_freq]

        a, b = most_freq
        # print(a, b)

        merges.append(most_freq)
        vocab.append(a+b)

        # in the pretokenized dict, replace keys (..., a, b, ...) with (..., ab, ...)
        for bs in list(pretokenized.keys()):

            pre_freq = pretokenized.get(bs, 0)

            i = 0
            while i < len(bs)-1:
                if bs[i:i+2] == most_freq:

                    # replace (..., a, b, ...) with (..., ab, ...)
                    new_bs = bs[:i] + (a+b,) + bs[i+2:]
                    pretokenized[new_bs] = pre_freq
                    del pretokenized[bs]

                    # update pair_freq

                    # add (prev, (a,b)) and ((a,b), next)
                    if i > 0:
                        pair_freq[bs[i-1], a+b] += pre_freq

                    if i < len(bs)-2:
                        pair_freq[a+b, bs[i+2]] += pre_freq

                    # remove (prev, a), (b, next)
                    if i > 0:
                        k = (bs[i-1], a)
                        pair_freq[k] -= pre_freq
                        if pair_freq[k] <= 0:
                            del pair_freq[k]
                    
                    if i < len(bs)-2:
                        k = (b, bs[i+2])
                        pair_freq[k] -= pre_freq
                        if pair_freq[k] <= 0:
                            del pair_freq[k]
                    
                    bs = new_bs
                else:
                    i += 1
    
    return {i: b for i, b in enumerate(vocab)}, merges


if __name__ == "__main__":
    # input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    input_path = "data/tokenizer_test.txt"
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<|endoftext|>"]
    vocab_size = 256 + len(special_tokens) + 12

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print(f'vocab size: {len(vocab)}')
    assert merges[-1] == (b'lowe', b'r')
