import tensorflow as tf
from conftest import NUM_TASKS, TRAIN_SIZE

from models import (
    AITM,
    MLP,
    MixtureOfExperts,
    MultiGateMixtureOfExperts,
    MultiTaskBCE,
    SharedBottom,
)


def test_mlp(mlp_model: MLP, sample_continuous_input: tf.Tensor) -> None:
    out = mlp_model(sample_continuous_input)

    assert out.shape == (TRAIN_SIZE, 1)


def test_shared_bottom(
    shared_bottom_model: SharedBottom, sample_categorical_input: tf.Tensor, sample_continuous_input: tf.Tensor
) -> None:
    out = shared_bottom_model([sample_categorical_input, sample_continuous_input])

    assert out.shape == (TRAIN_SIZE, NUM_TASKS)


def test_shared_bottom_training(shared_bottom_model: SharedBottom, sample_dataset: tf.data.Dataset) -> None:
    shared_bottom_model.compile(optimizer="adam", loss=MultiTaskBCE(NUM_TASKS))
    history = shared_bottom_model.fit(x=sample_dataset, epochs=1)

    assert len(history.history["loss"]) == 1


def test_mixtures_of_experts(
    moe_model: MixtureOfExperts, sample_categorical_input: tf.Tensor, sample_continuous_input: tf.Tensor
) -> None:
    out = moe_model([sample_categorical_input, sample_continuous_input])

    assert out.shape == (TRAIN_SIZE, NUM_TASKS)


def test_mixture_of_experts_training(moe_model: MixtureOfExperts, sample_dataset: tf.data.Dataset) -> None:
    moe_model.compile(optimizer="adam", loss=MultiTaskBCE(NUM_TASKS))
    history = moe_model.fit(x=sample_dataset, epochs=1)

    assert len(history.history["loss"]) == 1


def test_multigate_mixtures_of_experts(
    mgmoe_model: MultiGateMixtureOfExperts, sample_categorical_input: tf.Tensor, sample_continuous_input: tf.Tensor
) -> None:
    out = mgmoe_model([sample_categorical_input, sample_continuous_input])

    assert out.shape == (TRAIN_SIZE, NUM_TASKS)


def test_multigate_mixture_of_experts_training(
    mgmoe_model: MultiGateMixtureOfExperts, sample_dataset: tf.data.Dataset
) -> None:
    mgmoe_model.compile(optimizer="adam", loss=MultiTaskBCE(NUM_TASKS))
    history = mgmoe_model.fit(x=sample_dataset, epochs=1)

    assert len(history.history["loss"]) == 1


def test_aitm(aitm_model: AITM, sample_categorical_input: tf.Tensor, sample_continuous_input: tf.Tensor) -> None:
    out = aitm_model([sample_categorical_input, sample_continuous_input])

    assert out.shape == (TRAIN_SIZE, NUM_TASKS)


def test_aitm_training(aitm_model: AITM, sample_dataset: tf.data.Dataset) -> None:
    aitm_model.compile(optimizer="adam", loss=MultiTaskBCE(NUM_TASKS))
    history = aitm_model.fit(x=sample_dataset, epochs=1)

    assert len(history.history["loss"]) == 1
