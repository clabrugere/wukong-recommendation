from tensorflow import Tensor

from model.tensorflow.embedding import Embedding
from model.tensorflow.mlp import MLP
from model.tensorflow.wukong import (
    FactorizationMachineBlock,
    LinearCompressBlock,
    ResidualProjection,
    Wukong,
    WukongLayer,
)

from .conftest import BATCH_SIZE, DIM_EMB, NUM_CAT_FEATURES, NUM_DENSE_FEATURES


def test_mlp(mlp_model: MLP, sample_2d_continuous_input: Tensor) -> None:
    output = mlp_model(sample_2d_continuous_input)

    assert output.shape == (BATCH_SIZE, 16)


def test_embedding(
    embedding_layer: Embedding,
    sample_categorical_input: Tensor,
    sample_2d_continuous_input: Tensor,
) -> None:
    output = embedding_layer([sample_categorical_input, sample_2d_continuous_input])

    assert output.shape == (BATCH_SIZE, NUM_CAT_FEATURES + NUM_DENSE_FEATURES, DIM_EMB)


def test_lcb(lcb_layer: LinearCompressBlock, sample_3d_continuous_input: Tensor) -> None:
    output = lcb_layer(sample_3d_continuous_input)

    assert output.shape == (BATCH_SIZE, NUM_CAT_FEATURES // 2, DIM_EMB)


def test_fmb(fmb_layer: FactorizationMachineBlock, sample_3d_continuous_input: Tensor) -> None:
    output = fmb_layer(sample_3d_continuous_input)

    assert output.shape == (BATCH_SIZE, NUM_CAT_FEATURES // 2, DIM_EMB)


def test_residual_projection(residual_projection: ResidualProjection, sample_3d_continuous_input: Tensor) -> None:
    output = residual_projection(sample_3d_continuous_input)

    assert output.shape == (BATCH_SIZE, NUM_CAT_FEATURES // 2, DIM_EMB)


def test_residual_projection_identity(
    residual_projection_identity: ResidualProjection,
    sample_3d_continuous_input: Tensor,
) -> None:
    output = residual_projection_identity(sample_3d_continuous_input)

    assert output.shape == sample_3d_continuous_input.shape


def test_wukong_layer(wukong_layer: WukongLayer, sample_3d_continuous_input: Tensor) -> None:
    output = wukong_layer(sample_3d_continuous_input)

    assert output.shape == (BATCH_SIZE, 16 + 16, DIM_EMB)


def test_wukong_model(
    wukong_model: Wukong,
    sample_categorical_input: Tensor,
    sample_2d_continuous_input: Tensor,
) -> None:
    output = wukong_model([sample_categorical_input, sample_2d_continuous_input])

    assert output.shape == (BATCH_SIZE, 1)
