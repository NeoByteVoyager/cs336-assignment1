import os
from typing import BinaryIO
from collections import defaultdict
import regex as re
from multiprocessing import Pool

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


## Usage
def pre_tokenization(
        text: str,
        special_tokens: list[str]
):
    '''
    pre tokenization
    return:
        words_freq: dict[tuple(bytes), int]
    '''
    if special_tokens:
        PAT = "|".join(re.escape(token) for token in special_tokens)
        chunks = re.split(PAT, text)
    else:
        chunks = [text]

    words_freq = defaultdict(int)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        words = re.findall(PAT, chunk)
        for word in words:
            byte_word = tuple(bytes([b]) for b in word.encode("utf-8"))
            words_freq[byte_word] += 1
    return words_freq

def process_chunk(args):
    chunk, special_token = args
    return pre_tokenization(chunk, special_token)

def generate_chunks(
        input_path: str,
        num_processes: int,
):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            yield chunk

def multiprocess_tokenize(
    input_path: str,
    num_processes: int,
    special_tokens: list[str]
):
    # args for every chunk process
    tasks = ((chunk, special_tokens) for chunk in generate_chunks(input_path, num_processes))
    # multiprocess
    words_freq = defaultdict(int)
    with Pool(num_processes) as pool:
        for result in pool.imap(process_chunk, tasks):
            for words, freq in result.items():
                words_freq[words] += freq

    return words_freq

