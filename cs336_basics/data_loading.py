import numpy as np
import torch

def get_batch(
        dataset: np.array,
        batch_size: int,
        context_length: int,
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length

    starts = np.random.randint(0, max_start, batch_size)

    inputs = []
    targets = []

    for s in starts:
        inputs.append(dataset[s: s + context_length])
        targets.append(dataset[s + 1: s + context_length + 1])


    inputs = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
    targets = torch.tensor(np.array(targets), dtype=torch.long, device=device)

    return (inputs, targets)

if __name__ == "__main__":
    data = np.load("data/tinystories_tokens.npy")
    print(get_batch(data, 4, 8, "cpu"))