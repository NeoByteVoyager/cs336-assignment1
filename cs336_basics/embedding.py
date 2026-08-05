import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        '''
        num_embeddings(int): size of vocab
        embedding_dim(int): dimension of embedding dim
        '''
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype),
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        )

    def forward(self, input_ids):
        return self.weight[input_ids]


if __name__ == "__main__":
    input_id = torch.tensor([1,2,3], dtype=torch.long)
    embedding = Embedding(100, 5)
    print(embedding(input_id))