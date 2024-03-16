from typing import Any

from keras import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, ReLU


class MLP(Sequential):
    def __init__(
        self,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int | None = None,
        dropout: float = 0.0,
        name: str = "MLP",
    ) -> None:
        self.num_hidden = num_hidden
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.dropout = dropout

        layers = []
        for _ in range(num_hidden - 1):
            layers.append(Dense(dim_hidden, kernel_initializer="he_uniform"))
            layers.append(BatchNormalization())
            layers.append(ReLU())

            if 0.0 < dropout < 1.0:
                layers.append(Dropout(dropout))

        if dim_out:
            layers.append(Dense(dim_out, kernel_initializer="he_uniform"))
        else:
            layers.append(Dense(dim_hidden, kernel_initializer="he_uniform"))

        super().__init__(layers, name=name)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_hidden": self.num_hidden,
                "dim_hidden": self.dim_hidden,
                "dim_out": self.dim_out,
                "dropout": self.dropout,
                "name": self.name,
            }
        )

        return config
