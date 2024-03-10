import pytest
import torch

from pytorch.mlp import MLP
from pytorch.wukong import (
    FactorizationMachineBlock,
    LinearCompressBlock,
    ResidualProjection,
    Wukong,
    WukongLayer,
)

BATCH_SIZE = 16
NUM_FEATURES = 32
NUM_EMBEDDING = 100
DIM_EMB = 128


@pytest.fixture(scope="module")
def sample_2d_continuous_input():
    return torch.rand(BATCH_SIZE, NUM_FEATURES)


@pytest.fixture(scope="module")
def sample_3d_continuous_input():
    return torch.rand(BATCH_SIZE, NUM_FEATURES, DIM_EMB)


@pytest.fixture(scope="module")
def sample_categorical_input():
    return torch.multinomial(
        torch.rand((BATCH_SIZE, NUM_EMBEDDING)),
        NUM_FEATURES,
        replacement=True,
    )


@pytest.fixture
def mlp_model() -> MLP:
    return MLP(dim_in=NUM_FEATURES, num_hidden=3, dim_hidden=16, dim_out=1)


@pytest.fixture
def lcb_layer() -> LinearCompressBlock:
    return LinearCompressBlock(NUM_FEATURES, NUM_FEATURES // 2)


@pytest.fixture
def fmb_layer() -> FactorizationMachineBlock:
    return FactorizationMachineBlock(
        num_emb_in=NUM_FEATURES,
        num_emb_out=16,
        dim_emb=DIM_EMB,
        rank=4,
        num_hidden=2,
        dim_hidden=16,
        dropout=0.0,
    )


@pytest.fixture
def residual_projection() -> ResidualProjection:
    return ResidualProjection(NUM_FEATURES, NUM_FEATURES // 2)


@pytest.fixture
def residual_projection_identity() -> ResidualProjection:
    return ResidualProjection(NUM_FEATURES, NUM_FEATURES)


@pytest.fixture
def wukong_layer() -> WukongLayer:
    return WukongLayer(
        num_emb_in=NUM_FEATURES,
        dim_emb=DIM_EMB,
        num_emb_lcb=16,
        num_emb_fmb=16,
        rank_fmb=8,
        num_hidden=2,
        dim_hidden=16,
        dropout=0.0,
    )


@pytest.fixture
def wukong_model() -> Wukong:
    return Wukong(
        num_layers=2,
        num_emb=NUM_EMBEDDING,
        dim_emb=DIM_EMB,
        dim_input=NUM_FEATURES,
        num_emb_lcb=16,
        num_emb_fmb=16,
        rank_fmb=8,
        num_hidden_wukong=2,
        dim_hidden_wukong=16,
        num_hidden_head=2,
        dim_hidden_head=32,
        dim_output=1,
        dropout=0.0,
    )
