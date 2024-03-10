from torch import Tensor
from torch.nn import Module


def test_mlp(mlp_model: Module, sample_2d_continuous_input: Tensor) -> None:
    output = mlp_model(sample_2d_continuous_input)

    assert output.shape == (sample_2d_continuous_input.shape[0], 1)


def test_lcb(lcb_layer: Module, sample_3d_continuous_input: Tensor) -> None:
    output = lcb_layer(sample_3d_continuous_input)

    assert output.shape == (
        sample_3d_continuous_input.shape[0],
        sample_3d_continuous_input.shape[1] // 2,
        sample_3d_continuous_input.shape[2],
    )


def test_fmb(fmb_layer: Module, sample_3d_continuous_input: Tensor) -> None:
    output = fmb_layer(sample_3d_continuous_input)

    assert output.shape == (
        sample_3d_continuous_input.shape[0],
        sample_3d_continuous_input.shape[1] // 2,
        sample_3d_continuous_input.shape[2],
    )


def test_residual_projection(residual_projection: Module, sample_3d_continuous_input: Tensor) -> None:
    output = residual_projection(sample_3d_continuous_input)

    assert output.shape == (
        sample_3d_continuous_input.shape[0],
        sample_3d_continuous_input.shape[1] // 2,
        sample_3d_continuous_input.shape[2],
    )


def test_residual_projection_identity(
    residual_projection_identity: Module, sample_3d_continuous_input: Tensor
) -> None:
    output = residual_projection_identity(sample_3d_continuous_input)

    assert output.shape == sample_3d_continuous_input.shape


def test_wukong_layer(wukong_layer: Module, sample_3d_continuous_input: Tensor) -> None:
    output = wukong_layer(sample_3d_continuous_input)

    assert output.shape == (
        sample_3d_continuous_input.shape[0],
        16 + 16,
        sample_3d_continuous_input.shape[2],
    )


def test_wukong_model(wukong_model: Module, sample_categorical_input: Tensor) -> None:
    output = wukong_model(sample_categorical_input)

    assert output.shape == (sample_categorical_input.shape[0], 1)
