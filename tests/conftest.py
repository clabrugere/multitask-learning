import pytest
import tensorflow as tf

from models import AITM, MLP, MixtureOfExperts, MultiGateMixtureOfExperts, SharedBottom
from models.embedding import MultiInputEmbedding

TRAIN_SIZE = 320
NUM_TASKS = 2
NUM_CATEGORICAL = 10
NUM_CONTINUOUS = 10
NUM_EMBEDDING = 100


@pytest.fixture(scope="module")
def sample_categorical_input() -> tf.Tensor:
    return tf.random.categorical(
        tf.random.uniform((TRAIN_SIZE, NUM_EMBEDDING), minval=0, maxval=1), NUM_CATEGORICAL, dtype=tf.int32
    )


@pytest.fixture(scope="module")
def sample_continuous_input() -> tf.Tensor:
    return tf.random.uniform(shape=(TRAIN_SIZE, NUM_CONTINUOUS), dtype=tf.float32)


@pytest.fixture(scope="module")
def sample_dataset() -> tf.data.Dataset:
    inputs = (
        tf.random.categorical(
            tf.random.uniform((TRAIN_SIZE, NUM_EMBEDDING), minval=0, maxval=1), NUM_CATEGORICAL, dtype=tf.int32
        ),
        tf.random.uniform(shape=(TRAIN_SIZE, NUM_CONTINUOUS), dtype=tf.float32),
    )
    labels = tf.cast(tf.random.uniform(shape=(TRAIN_SIZE, NUM_TASKS)) > 0.5, tf.int32)

    return tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(32).prefetch(tf.data.AUTOTUNE)


@pytest.fixture
def embedding_layer() -> MultiInputEmbedding:
    return MultiInputEmbedding(num_categorical_emb=NUM_EMBEDDING, dim_emb=8)


@pytest.fixture
def mlp_model() -> MLP:
    return MLP(num_hidden=2, dim_hidden=16, dim_out=1)


@pytest.fixture
def shared_bottom_model() -> SharedBottom:
    return SharedBottom(
        num_tasks=NUM_TASKS,
        num_emb=NUM_EMBEDDING,
        dim_emb=8,
        embedding_l2=0.0,
        num_hidden_shared=2,
        dim_hidden_shared=16,
        dropout_shared=0.0,
        num_hidden_tasks=2,
        dim_hidden_tasks=16,
        dim_out_tasks=1,
        dropout_tasks=0.0,
    )


@pytest.fixture
def moe_model() -> MixtureOfExperts:
    return MixtureOfExperts(
        num_tasks=NUM_TASKS,
        num_emb=NUM_EMBEDDING,
        dim_emb=8,
        num_experts=2,
        num_hidden_expert=2,
        dim_hidden_expert=16,
        dropout_expert=0.0,
        gate_function="softmax",
        num_hidden_tasks=2,
        dim_hidden_tasks=16,
        dim_out_tasks=1,
        dropout_tasks=0.0,
    )


@pytest.fixture
def mgmoe_model() -> MultiGateMixtureOfExperts:
    return MultiGateMixtureOfExperts(
        num_tasks=NUM_TASKS,
        num_emb=NUM_EMBEDDING,
        dim_emb=8,
        num_experts=2,
        num_hidden_expert=2,
        dim_hidden_expert=16,
        dropout_expert=0.0,
        gate_function="softmax",
        num_hidden_tasks=2,
        dim_hidden_tasks=16,
        dim_out_tasks=1,
        dropout_tasks=0.0,
    )


@pytest.fixture
def aitm_model() -> AITM:
    return AITM(
        num_tasks=NUM_TASKS,
        num_emb=NUM_EMBEDDING,
        dim_emb=8,
        embedding_l2=0.0,
        num_hidden_tower=2,
        dim_hidden_tower=16,
        dropout_tower=0.0,
        num_hidden_head=2,
        dim_hidden_head=16,
        dropout_proj_head=0.0,
        dim_out_tasks=1,
    )


@pytest.fixture
def model(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def input(request):
    return request.getfixturevalue(request.param)
