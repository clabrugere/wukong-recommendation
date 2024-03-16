from typing import Any

import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Identity, Layer, LayerNormalization
from tensorflow import Tensor, TensorShape

from model.tensorflow.embedding import Embedding
from model.tensorflow.mlp import MLP


class LinearCompressBlock(Layer):
    def __init__(self, num_emb_out: int, weights_initializer: str = "he_uniform", name: str = "lcb") -> None:
        super().__init__(name=name)
        self.num_emb_out = num_emb_out
        self.weights_initializer = weights_initializer

    def build(self, input_shape: TensorShape) -> None:
        num_emb_in = input_shape[1]

        self.weight = self.add_weight(
            name="weight",
            shape=(num_emb_in, self.num_emb_out),
            initializer=self.weights_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = tf.transpose(inputs, (0, 2, 1))

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        outputs = outputs @ self.weight

        # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        outputs = tf.transpose(outputs, (0, 2, 1))

        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_emb_out": self.num_emb_out,
                "weights_initializer": self.weights_initializer,
            }
        )

        return config


class FactorizationMachineBlock(Layer):
    def __init__(
        self,
        num_emb_out: int,
        dim_emb: int,
        rank: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
        weights_initializer: str = "he_uniform",
        name: str = "fmb",
    ) -> None:
        super().__init__(name=name)

        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank
        self.weights_initializer = weights_initializer

        self.norm = LayerNormalization()
        self.mlp = MLP(
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=num_emb_out * dim_emb,
            dropout=dropout,
        )

    def build(self, input_shape: TensorShape) -> None:
        self.num_emb_in = input_shape[1]

        self.weight = self.add_weight(
            name="weight",
            shape=(self.num_emb_in, self.rank),
            initializer=self.weights_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = tf.transpose(inputs, (0, 2, 1))

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, rank) -> (bs, dim_emb, rank)
        outputs = outputs @ self.weight

        # (bs, num_emb_in, dim_emb) @ (bs, dim_emb, rank) -> (bs, num_emb_in, rank)
        outputs = inputs @ outputs

        # (bs, num_emb_in, rank) -> (bs, num_emb_in * rank)
        outputs = tf.reshape(outputs, (-1, self.num_emb_in * self.rank))

        # (bs, num_emb_in * rank) -> (bs, num_emb_out * dim_emb)
        outputs = self.mlp(self.norm(outputs))

        # (bs, num_emb_out * dim_emb) -> (bs, num_emb_out, dim_emb)
        outputs = tf.reshape(outputs, (-1, self.num_emb_out, self.dim_emb))

        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_emb_out": self.num_emb_out,
                "dim_emb": self.dim_emb,
                "rank": self.rank,
                "weights_initializer": self.weights_initializer,
            }
        )

        return config


class ResidualProjection(Layer):
    def __init__(
        self,
        num_emb_out: int,
        weights_initializer: str = "he_uniform",
        name: str = "residual_projection",
    ) -> None:
        super().__init__(name=name)

        self.num_emb_out = num_emb_out
        self.weights_initializer = weights_initializer

    def build(self, input_shape: TensorShape) -> None:
        self.num_emb_in = input_shape[1]

        self.weight = self.add_weight(
            name="weight",
            shape=(self.num_emb_in, self.num_emb_out),
            initializer=self.weights_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        self.built = True

    def call(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = tf.transpose(inputs, (0, 2, 1))

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        outputs = outputs @ self.weight

        # # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        outputs = tf.transpose(outputs, (0, 2, 1))

        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_emb_out": self.num_emb_out,
                "weights_initializer": self.weights_initializer,
            }
        )

        return config


class WukongLayer(Layer):
    def __init__(
        self,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
        name: str = "wukong",
    ) -> None:
        super().__init__(name=name)

        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        self.rank_fmb = rank_fmb
        self.num_hidden = num_hidden
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        self.norm = LayerNormalization()

    def build(self, input_shape: TensorShape) -> None:
        num_emb_in, dim_emb = input_shape[-2:]

        self.lcb = LinearCompressBlock(self.num_emb_lcb)
        self.fmb = FactorizationMachineBlock(
            self.num_emb_fmb,
            dim_emb,
            self.rank_fmb,
            self.num_hidden,
            self.dim_hidden,
            self.dropout,
        )

        if num_emb_in != self.num_emb_lcb + self.num_emb_fmb:
            self.residual_projection = ResidualProjection(self.num_emb_lcb + self.num_emb_fmb)
        else:
            self.residual_projection = Identity()

        self.built = True

    def call(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_lcb, dim_emb)
        lcb = self.lcb(inputs)

        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_fmb, dim_emb)
        fmb = self.fmb(inputs)

        # (bs, num_emb_lcb, dim_emb), (bs, num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = tf.concat((fmb, lcb), axis=1)

        # (bs, num_emb_lcb + num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = self.norm(outputs + self.residual_projection(inputs))

        return outputs

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_emb_lcb": self.num_emb_lcb,
                "num_emb_fmb": self.num_emb_fmb,
                "rank_fmb": self.rank_fmb,
                "num_hidden": self.num_hidden,
                "dim_hidden": self.dim_hidden,
                "dropout": self.dropout,
            }
        )

        return config


class Wukong(Model):
    def __init__(
        self,
        num_layers: int,
        num_sparse_emb: int,
        dim_emb: int,
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

        self.embedding = Embedding(num_sparse_emb, dim_emb)

        self.interaction_layers = Sequential()
        for i in range(num_layers):
            self.interaction_layers.add(
                WukongLayer(
                    num_emb_lcb,
                    num_emb_fmb,
                    rank_fmb,
                    num_hidden_wukong,
                    dim_hidden_wukong,
                    dropout,
                    name=f"wukong_{i}",
                ),
            )

        self.projection_head = MLP(
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
        )

    def call(self, inputs: list[Tensor]) -> Tensor:
        outputs = self.embedding(inputs)
        outputs = self.interaction_layers(outputs)
        outputs = tf.reshape(outputs, (-1, (self.num_emb_lcb + self.num_emb_fmb) * self.dim_emb))
        outputs = self.projection_head(outputs)

        return outputs
