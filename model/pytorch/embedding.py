import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, num_emb: int, dim_emb: int, num_dense: int, bias: bool = True) -> None:
        super().__init__()

        self.dim_emb = dim_emb
        self.num_dense = num_dense

        self.sparse_embedding = nn.Embedding(num_emb, dim_emb)
        self.dense_embedding = nn.Linear(num_dense, num_dense * dim_emb, bias=bias)

    def forward(self, sparse_inputs: Tensor, dense_inputs) -> Tensor:
        sparse_outputs = self.sparse_embedding(sparse_inputs)
        dense_outputs = self.dense_embedding(dense_inputs).view(-1, self.num_dense, self.dim_emb)

        return torch.cat([sparse_outputs, dense_outputs], dim=1)
