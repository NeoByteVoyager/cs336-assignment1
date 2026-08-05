from collections import defaultdict
import regex as re
from .pretokenization import multiprocess_tokenize
import heapq

class Pair:
    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

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
    pair_freq = defaultdict(int)
    for word, freq in words_freq.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freq[pair] += freq

    heap = []
    for pair, freq in pair_freq.items():
            heapq.heappush(heap, (-freq, Pair(pair)))

    while len(vocab) < vocab_size:

        # max pair freq
        max_freq_pair = None
        while heap:
            neg_freq, pair_obj = heapq.heappop(heap)
            freq = - neg_freq
            pair = pair_obj.pair

            if freq == pair_freq[pair]:
                max_freq_pair = pair
                break

        if not max_freq_pair:
            break

        vocab[len(vocab)] = max_freq_pair[0] + max_freq_pair[1]
        merges.append(max_freq_pair)

        # merge
        tmp_freq  = defaultdict(int)
        affected_pairs = set()
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

            new_words = tuple(new_words)
            tmp_freq[new_words] += freq
            if word != new_words:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_freq[pair] -= freq
                    affected_pairs.add(pair)
                for i in range(len(new_words) - 1):
                    pair = (new_words[i], new_words[i + 1])
                    pair_freq[pair] += freq
                    affected_pairs.add(pair)
        words_freq = tmp_freq

        for pair in affected_pairs:
            if pair_freq[pair]:
                heapq.heappush(heap, (-pair_freq[pair], Pair(pair)))

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

    '''
    words_freq = multiprocess_tokenize(
        "data/tinystories.txt",
        8,
        ["<|endoftext|>"]
    )
    '''
    end = time.time()

    print("time:", end-start)



