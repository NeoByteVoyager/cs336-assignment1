from typing import Iterable, Iterator
import json
import regex as re
import numpy as np

def pre_tokenizer(text: str, special_tokens: list[str] | None):
    '''
    pre tokenization
    return:
         list[tuple(bytes) | str]
    '''

    if special_tokens:
        PAT = "(" + "|".join(sorted(
            [re.escape(token) for token in special_tokens],
            key=len,
            reverse=True
        )
        ) + ")"

        chunks = re.split(PAT, text)
    else:
        chunks = [text]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words = []
    for chunk in chunks:
        if chunk in special_tokens:
            words.append(chunk)
            continue
        for word in re.findall(PAT, chunk):
            words.append(tuple(bytes([b]) for b in word.encode("utf-8")))
    return words


class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple],
                 special_token: list[str] | None,
                 ):
        # two-way table
        self.id_to_token = vocab
        self.token_to_id = {value: key for key, value in vocab.items()}
        # merge and merge_rank
        self.merge_rank = {
            merge: i
            for i, merge in enumerate(merges)
        }
        # cache to store already token_to_id
        self.cache = {}
        # special_token
        self.special_token = special_token or []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        # load vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        # load merges
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_data = json.load(f)

        # restore vocab
        vocab = {
            int(idx): bytes.fromhex(token)
            for idx, token in vocab_data.items()
        }

        # restore merges
        merges = [
            (bytes.fromhex(a), bytes.fromhex(b))
            for a, b in merges_data
        ]

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        words = pre_tokenizer(text, self.special_token)
        res = []
        for word in words:
            if isinstance(word, str):
                res.append(self.token_to_id[word.encode("utf-8")])
                continue
            if word in self.cache:
                res.extend(self.cache[word])
                continue
            original_word = word

            while True:
                # Find the min pair
                merge = None
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    if pair in self.merge_rank:
                        if merge == None or self.merge_rank[pair] < self.merge_rank[merge]:
                            merge = pair
                if merge == None:
                    break
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == merge:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
            ids = []
            for token in word:
                ids.append(self.token_to_id[token])
            res.extend(ids)
            self.cache[original_word] = ids
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        return b"".join(
            self.id_to_token[id] for id in ids
        ).decode("utf-8", errors="replace")


'''
vocab = {0: b' ', 1: b'a', 2:b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at', 11: b'<|endoftext|>', 12:b'<|endoftext|><|endoftext|>'}
merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a',b't')]
special_token = ["<|endoftext|>","<|endoftext|><|endoftext|>"]
tokenizer = Tokenizer(vocab, merges, special_token)

enco = tokenizer.encode("the cat<|endoftext|><|endoftext|> ate<|endoftext|>")
print(enco)
'''

if __name__ == "__main__":
    tokenizer = Tokenizer.from_files("data/tokenizer_data/vocab.json","data/tokenizer_data/merges.json", ["<|endoftext|>"])
    s = "\nOnce upon a time there was a little boy named Ben.\n"
    enc = tokenizer.encode(s)
    print(enc)
    ''' tokenize tiny_stories
    ids = []
    with open("data/tinystories.txt", "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)

    ids = np.array(ids, dtype=np.uint16)

    np.save("data/tinystories_tokens.npy", ids)
    '''