import torch
from torch import Tensor, nn

from model.pytorch.embedding import Embedding
from model.pytorch.mlp import MLP


class LinearCompressBlock(nn.Module):
    def __init__(self, num_emb_in: int, num_emb_out: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty((num_emb_in, num_emb_out)))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = inputs.permute(0, 2, 1)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        outputs = outputs @ self.weight

        # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.permute(0, 2, 1)

        return outputs


class FactorizationMachineBlock(nn.Module):
    def __init__(
        self,
        num_emb_in: int,
        num_emb_out: int,
        dim_emb: int,
        rank: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank

        self.weight = nn.Parameter(torch.empty((num_emb_in, rank)))
        self.norm = nn.LayerNorm(num_emb_in * rank)
        self.mlp = MLP(
            dim_in=num_emb_in * rank,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=num_emb_out * dim_emb,
            dropout=dropout,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = inputs.permute(0, 2, 1)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, rank) -> (bs, dim_emb, rank)
        outputs = outputs @ self.weight

        # (bs, num_emb_in, dim_emb) @ (bs, dim_emb, rank) -> (bs, num_emb_in, rank)
        outputs = torch.bmm(inputs, outputs)

        # (bs, num_emb_in, rank) -> (bs, num_emb_in * rank)
        outputs = outputs.view(-1, self.num_emb_in * self.rank)

        # (bs, num_emb_in * rank) -> (bs, num_emb_out * dim_emb)
        outputs = self.mlp(self.norm(outputs))

        # (bs, num_emb_out * dim_emb) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.view(-1, self.num_emb_out, self.dim_emb)

        return outputs


class ResidualProjection(nn.Module):
    def __init__(self, num_emb_in: int, num_emb_out: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty((num_emb_in, num_emb_out)))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = inputs.permute(0, 2, 1)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        outputs = outputs @ self.weight

        # # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.permute(0, 2, 1)

        return outputs


class WukongLayer(nn.Module):
    def __init__(
        self,
        num_emb_in: int,
        dim_emb: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.lcb = LinearCompressBlock(num_emb_in, num_emb_lcb)
        self.fmb = FactorizationMachineBlock(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
        )
        self.norm = nn.LayerNorm(dim_emb)

        if num_emb_in != num_emb_lcb + num_emb_fmb:
            self.residual_projection = ResidualProjection(num_emb_in, num_emb_lcb + num_emb_fmb)
        else:
            self.residual_projection = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_lcb, dim_emb)
        lcb = self.lcb(inputs)

        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_fmb, dim_emb)
        fmb = self.fmb(inputs)

        # (bs, num_emb_lcb, dim_emb), (bs, num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = torch.concat((fmb, lcb), dim=1)

        # (bs, num_emb_lcb + num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = self.norm(outputs + self.residual_projection(inputs))

        return outputs


class Wukong(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_sparse_emb: int,
        dim_emb: int,
        dim_input_sparse: int,
        dim_input_dense: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        num_hidden_wukong: int,
        dim_hidden_wukong: int,
        num_hidden_head: int,
        dim_hidden_head: int,
        dim_output: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.dim_emb = dim_emb
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb

        self.embedding = Embedding(num_sparse_emb, dim_emb, dim_input_dense)

        num_emb_in = dim_input_sparse + dim_input_dense
        self.interaction_layers = nn.Sequential()
        for _ in range(num_layers):
            self.interaction_layers.append(
                WukongLayer(
                    num_emb_in,
                    dim_emb,
                    num_emb_lcb,
                    num_emb_fmb,
                    rank_fmb,
                    num_hidden_wukong,
                    dim_hidden_wukong,
                    dropout,
                ),
            )
            num_emb_in = num_emb_lcb + num_emb_fmb

        self.projection_head = MLP(
            (num_emb_lcb + num_emb_fmb) * dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
        )

    def forward(self, sparse_inputs: Tensor, dense_inputs) -> Tensor:
        outputs = self.embedding(sparse_inputs, dense_inputs)
        outputs = self.interaction_layers(outputs)
        outputs = outputs.view(-1, (self.num_emb_lcb + self.num_emb_fmb) * self.dim_emb)
        outputs = self.projection_head(outputs)

        return outputs
