import torch


def crossEntropy(inputs: torch.Tensor, ouputs: torch.Tensor):
    '''
    inputs: "batch vocab"
    outputs: "batch"
    '''
    ids = torch.arange(inputs.shape[0], device=inputs.device)
    correct = inputs[ids, ouputs]

    s = torch.logsumexp(inputs, -1)

    return (s - correct).mean()