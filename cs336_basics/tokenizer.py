import regex as re
from collections import defaultdict


def pre_tokenizer(text: str, special_tokens: list[str]):
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
        if chunk in special_token:
            words.append(chunk)
            continue
        for word in re.findall(PAT, chunk):
            words.append(tuple(bytes([b]) for b in word.encode("utf-8")))
    return words


class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple],
                 special_token: list[str],
    ):
        # 建立双向表
        self.id_to_token = vocab
        self.token_to_id = {value: key for key, value in vocab.items()}

        self.merges = merges
        self.special_token = special_token

    def encode(self, text: str) -> list[int]:
        words = pre_tokenizer(text, self.special_token)
        res = []
        for word in words:
            if isinstance(word, str):
                res.append(self.token_to_id[word.encode("utf-8")])
                continue
            for merge in self.merges:
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
            for token in word:
                res.append(self.token_to_id[token])
        return res

    def decode(self, ids: list[int]) -> str:
        return b"".join(
             self.id_to_token[id] for id in ids
        ).decode("utf-8", errors="replace")


vocab = {0: b' ', 1: b'a', 2:b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at', 11: b'<|endoftext|>', 12:b'<|endoftext|><|endoftext|>'}
merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a',b't')]
special_token = ["<|endoftext|>","<|endoftext|><|endoftext|>"]
tokenizer = Tokenizer(vocab, merges, special_token)
enco = tokenizer.encode("the cat<|endoftext|><|endoftext|> ate<|endoftext|>")
print(enco)

print(tokenizer.decode(enco))