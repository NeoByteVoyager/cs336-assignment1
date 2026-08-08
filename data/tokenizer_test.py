import json
from cs336_basics.train_bpe import train_bpe

from pathlib import Path
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]

    input_path = ROOT / "data" / "tinystories.txt"

    vocab_size = 10000

    special_tokens = ["<|endoftext|>"]


    # 训练BPE
    vocab, merges = train_bpe(
        input_path,
        vocab_size,
        special_tokens
    )


    # 保存目录
    save_dir = "tokenizer_data"


    import os
    os.makedirs(save_dir, exist_ok=True)


    # 保存 vocab
    # bytes不能直接json，所以转成utf-8字符串
    vocab_save = {
        str(k): v.hex()
        for k, v in vocab.items()
    }

    with open(f"{save_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_save, f, indent=2)


    # 保存 merges
    merges_save = [
        [
            a.hex(),
            b.hex()
        ]
        for a, b in merges
    ]


    with open(f"{save_dir}/merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_save, f,  indent=2)


    print("save tokenizer success")
    print("vocab size:", len(vocab))
    print("merge size:", len(merges))