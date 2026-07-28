from collections import defaultdict
import regex as re
from pretokenization_example import multiprocess_tokenize

def merge(
        vocab_size: int,
        words_freq: dict,
        vocab: dict,
        merges: list
):
    '''
    merge the most freq bytes
    returns:
        vocab: dict[int, bytes]
        merge: list[tuple[bytes, bytes]]
    '''
    while len(vocab) < vocab_size:
        # pair freq
        pair_freq = defaultdict(int)
        for word, freq in words_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] += freq

        if not pair_freq:
            break

        # max pair freq
        max_freq_pair = max(pair_freq, key=lambda x:(pair_freq[x], x))
        vocab[len(vocab)] = max_freq_pair[0] + max_freq_pair[1]
        merges.append(max_freq_pair)
        # merge
        tmp_freq  = defaultdict(int)
        for word, freq in words_freq.items():
            new_words = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == max_freq_pair:
                    new_words.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_words.append(word[i])
                    i += 1
            tmp_freq[tuple(new_words)] += freq

        words_freq = tmp_freq
    return vocab, merges

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
):
    '''
    train a byte-level BPE tokeni zer
    returns:
        vocab: dict[int, bytes]
        merge: list[tuple[bytes, bytes]]
    '''
    # Initialize
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    # Add special_token
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # Pre_tokenization
    words_freq = multiprocess_tokenize(input_path, 8, special_tokens)

    # Merge
    return merge(vocab_size, words_freq, vocab, merges)


if __name__ == "__main__":
    import time


    start = time.time()

    vocab, merges = train_bpe(
        "data/tinystories.txt",
        500,
        ["<|endoftext|>"]
    )

    end = time.time()

    print("pre_tokenization time:", end-start)



