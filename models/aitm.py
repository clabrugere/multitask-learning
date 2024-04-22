import tensorflow as tf
from keras import Model, activations
from keras.layers import Attention, Dense, Flatten, Layer

from models.embedding import MultiInputEmbedding
from models.mlp import MLP


class AIT(Layer):
    def __init__(self, dim_hidden: int, dropout: float = 0.0, name="ait") -> None:
        super().__init__(name=name)

        self.query = Dense(dim_hidden, use_bias=False)
        self.key = Dense(dim_hidden, use_bias=False)
        self.value = Dense(dim_hidden, use_bias=False)
        self.attention = Attention(dropout=dropout)

    def build(self, input_shape):
        self.query.build(input_shape)
        self.key.build(input_shape)
        self.value.build(input_shape)
        self.attention.build(input_shape)

    def call(self, previous: tf.Tensor, current: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        z = tf.concat((tf.expand_dims(previous, 1), tf.expand_dims(current, 1)), axis=1)  # (bs, 2, dim_hidden)

        query = self.query(z, training=training)  # (bs, 2, dim_hidden)
        key = self.key(z, training=training)  # (bs, 2, dim_hidden)
        value = self.value(z, training=training)  # (bs, 2, dim_hidden)

        out = self.attention([query, value, key], training=training)  # (bs, 2, dim_hidden)

        return tf.reduce_sum(out, axis=1)  # (bs, dim_hidden)


class AITM(Model):
    def __init__(
        self,
        num_tasks: int,
        num_emb: int,
        dim_emb: int = 32,
        embedding_l2: float = 0.0,
        num_hidden_tower: int = 2,
        dim_hidden_tower: int = 64,
        dropout_tower: float = 0.0,
        num_hidden_head: int = 2,
        dim_hidden_head: int = 64,
        dropout_proj_head: float = 0.0,
        dim_out_tasks: int = 1,
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.embedding = MultiInputEmbedding(
            num_categorical_emb=num_emb,
            dim_emb=dim_emb,
            regularization=embedding_l2,
        )
        self.flatten = Flatten()

        # the transfer learning mechanism between two subsequent tasks is basically self-attention between the latent
        # representations of two consecutive tasks.
        self.ait = AIT(dim_hidden=dim_hidden_tower)

        self.towers = []
        self.proj_heads = []
        self.gates = []
        for i in range(num_tasks):
            self.towers.append(MLP(num_hidden_tower, dim_hidden_tower, dropout=dropout_tower))
            self.proj_heads.append(MLP(num_hidden_head, dim_hidden_head, dim_out_tasks, dropout=dropout_proj_head))
            if i < num_tasks - 1:
                self.gates.append(Dense(dim_hidden_tower))

    def call(self, inputs: list[tf.Tensor], training: bool | None = None) -> tf.Tensor:
        sparse_inputs, dense_inputs = inputs

        embeddings = self.embedding([sparse_inputs, dense_inputs])
        embeddings = self.flatten(embeddings)

        q = [tower(embeddings, training=training) for tower in self.towers]  # [(bs, dim_hidden_tower), ...]
        for i in range(1, self.num_tasks):
            p = self.gates[i - 1](q[i - 1])  # (bs, dim_hidden_tower)
            q[i] = self.ait(p, q[i], training=training)  # (bs, dim_hidden_tower)

        out = tf.concat([head(x) for x, head in zip(q, self.proj_heads)], axis=-1)  # (bs, num_tasks)
        out = activations.sigmoid(out)  # (bs, num_tasks)

        return out
