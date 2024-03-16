from typing import Any

import tensorflow as tf
from keras.layers import Dense, Layer
from keras.layers import Embedding as SparseEmbedding
from tensorflow import Tensor, TensorShape


class Embedding(Layer):
    def __init__(self, num_sparse_emb: int, dim_emb: int, bias: bool = True) -> None:
        super().__init__()

        self.num_sparse_emb = num_sparse_emb
        self.dim_emb = dim_emb
        self.bias = bias

    def build(self, inputs_shape: list[TensorShape]) -> None:
        _, input_shape_dense = inputs_shape
        dim_dense = input_shape_dense[-1]

        self.sparse_embedding = SparseEmbedding(self.num_sparse_emb, self.dim_emb)
        self.dense_embedding = Dense(dim_dense * self.dim_emb, use_bias=self.bias)
        self.dim_dense = dim_dense
        self.built = True

    def call(self, inputs: list[Tensor]) -> Tensor:
        inputs_sparse, inputs_dense = inputs
        sparse_outputs = self.sparse_embedding(inputs_sparse)
        dense_outputs = tf.reshape(self.dense_embedding(inputs_dense), (-1, self.dim_dense, self.dim_emb))

        return tf.concat([sparse_outputs, dense_outputs], axis=1)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_sparse_emb": self.num_sparse_emb,
                "dim_emb": self.dim_emb,
                "bias": self.bias,
            }
        )

        return config
