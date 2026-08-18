import torch
import json
from cs336_basics.softmax import softmax
from cs336_basics.transformer_lm import Model
from cs336_basics.tokenizer import Tokenizer

# input an English seq, output a seq end with endtext or up to the max seq len
# model
model_config = {
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1344,
    "rope_theta": 10000
}
model = Model(**model_config)
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("checkpoints/batch64_lr0_002.pt", map_location=torch.device(device))
# load model and set eval mode
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


# tokenizer
with open("data/tokenizer_data/vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
vocab = {
    int(id): bytes.fromhex(token)
    for id, token in vocab_data.items()
}
with open("data/tokenizer_data/merges.json", "r", encoding="utf-8") as f:
    merges_data = json.load(f)
merges = [
    (bytes.fromhex(a), bytes.fromhex(b))
    for a, b in merges_data
]
tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

# input
inputs = input()
input_ids = tokenizer.encode(inputs)

i = 0
max_len = 240
t = 0.3
p = 0.8
end_id = tokenizer.encode("<|endoftext|>")[-1]


while i < max_len and input_ids[-1] != end_id:
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
    # probs (batch, seq, vocab)
    probs = softmax(logits / t, -1)
    # The last token probs (batch, vocab)
    next_token_probs = probs[:, -1, :]
    next_token_probs, ids = torch.sort(next_token_probs, dim=-1, descending=True)
    # prefix_sum
    cum_probs = torch.cumsum(next_token_probs, -1)
    # mask
    mask = cum_probs >= p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    # new probs
    next_token_probs = torch.masked_fill(next_token_probs, mask, 0)
    next_token_probs = next_token_probs / next_token_probs.sum()
    # sample
    sample_id = torch.multinomial(next_token_probs, num_samples=1).item()
    token_id = ids[0, sample_id].item()

    input_ids.append(token_id)
    i += 1

print(tokenizer.decode(input_ids))
