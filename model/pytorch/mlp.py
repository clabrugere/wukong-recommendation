from torch import nn


class MLP(nn.Sequential):
    def __init__(
        self,
        dim_in: int,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int | None = None,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        layers = []
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(dim_in, dim_hidden))

            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_hidden))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim_in = dim_hidden

        if dim_out:
            layers.append(nn.Linear(dim_in, dim_out))
        else:
            layers.append(nn.Linear(dim_in, dim_hidden))

        super().__init__(*layers)
