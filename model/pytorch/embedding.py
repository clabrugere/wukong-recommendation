import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, num_sparse_emb: int, dim_emb: int, dim_input_dense: int, bias: bool = True) -> None:
        super().__init__()

        self.dim_emb = dim_emb
        self.dim_input_dense = dim_input_dense

        self.sparse_embedding = nn.Embedding(num_sparse_emb, dim_emb)
        self.dense_embedding = nn.Linear(dim_input_dense, dim_input_dense * dim_emb, bias=bias)

    def forward(self, sparse_inputs: Tensor, dense_inputs) -> Tensor:
        sparse_outputs = self.sparse_embedding(sparse_inputs)
        dense_outputs = self.dense_embedding(dense_inputs).view(-1, self.dim_input_dense, self.dim_emb)

        return torch.concat((sparse_outputs, dense_outputs), dim=1)
