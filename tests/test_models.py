import pytest
import tensorflow as tf
from conftest import NUM_TASKS, TRAIN_SIZE
from tensorflow import Tensor
from tensorflow.python.keras import Model

from models import MLP, MultiTaskBCE


def test_mlp_output_shape(mlp_model: MLP, sample_continuous_input: tf.Tensor) -> None:
    out = mlp_model(sample_continuous_input)

    assert out.shape == (TRAIN_SIZE, 1)


@pytest.mark.parametrize(
    "model, sample_categorical_input, sample_continuous_input",
    [
        ("shared_bottom_model", "sample_categorical_input", "sample_continuous_input"),
        ("moe_model", "sample_categorical_input", "sample_continuous_input"),
        ("mgmoe_model", "sample_categorical_input", "sample_continuous_input"),
        ("aitm_model", "sample_categorical_input", "sample_continuous_input"),
    ],
    indirect=True,
)
def test_model_output_shape(model: Model, sample_categorical_input: Tensor, sample_continuous_input: Tensor) -> None:
    out = model([sample_categorical_input, sample_continuous_input])

    assert out.shape == (TRAIN_SIZE, NUM_TASKS)


@pytest.mark.parametrize(
    "model, sample_dataset",
    [
        ("shared_bottom_model", "sample_dataset"),
        ("moe_model", "sample_dataset"),
        ("mgmoe_model", "sample_dataset"),
        ("aitm_model", "sample_dataset"),
    ],
    indirect=True,
)
def test_model_training(model: Model, sample_dataset: tf.data.Dataset) -> None:
    model.compile(optimizer="adam", loss=MultiTaskBCE(NUM_TASKS))
    history = model.fit(x=sample_dataset, epochs=1)

    assert len(history.history["loss"]) == 1
