import torch
import torch.nn.functional as F

def softmax(x: torch.Tensor, dim:int):
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - max_val

    exp_x = torch.exp(x)

    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    print(x)
    print(softmax(x, -1))
    print(F.softmax(x, -1))