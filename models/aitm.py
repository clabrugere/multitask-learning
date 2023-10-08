import tensorflow as tf
from models.mlp import MLP


class AIT(tf.keras.layers.Layer):
    def __init__(self, dim_hidden, dropout=0.0):
        super().__init__()

        self.query = tf.keras.layers.Dense(dim_hidden)
        self.key = tf.keras.layers.Dense(dim_hidden)
        self.value = tf.keras.layers.Dense(dim_hidden)
        self.attention = tf.keras.layers.Attention(dropout=dropout)

    def build(self, input_shape):
        self.query.build(input_shape)
        self.key.build(input_shape)
        self.value.build(input_shape)
        self.attention.build(input_shape)

    def call(self, previous, current, training=None):
        z = tf.concat((tf.expand_dims(previous, 1), tf.expand_dims(current, 1)), axis=1)  # (bs, 2, dim_hidden)

        query = self.query(z, training=training)  # (bs, 2, dim_hidden)
        key = self.query(z, training=training)  # (bs, 2, dim_hidden)
        value = self.query(z, training=training)  # (bs, 2, dim_hidden)

        out = self.attention(query, value, key, training=training)  # (bs, 2, dim_hidden)

        return tf.reduce_sum(out, axis=1)  # (bs, dim_hidden)


class AITM(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_tasks,
        dim_continuous,
        num_emb,
        dim_emb=32,
        embedding_l2=0.0,
        num_hidden_tower=2,
        dim_hidden_tower=64,
        dropout_tower=0.0,
        num_hidden_head=2,
        dim_hidden_head=64,
        dropout_proj_head=0.0,
        dim_out_tasks=1,
    ):
        super().__init__()

        self.dim_input = dim_input
        self.num_tasks = num_tasks
        self.dim_emb = dim_emb

        self.continuous_proj = tf.keras.layers.Dense(dim_continuous)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_emb,
            output_dim=dim_emb,
            input_length=dim_input,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=embedding_l2),
        )

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
                self.gates.append(tf.keras.layers.Dense(dim_hidden_tower))

    def call(self, dense_inputs, discrete_inputs, training=None):
        emb_continuous = self.continuous_proj(dense_inputs, training=training)  # (bs, dim_continuous)
        emb_discrete = self.embedding(discrete_inputs, training=training)  # (bs, dim_input, dim_emb)
        emb_discrete = tf.reshape(emb_discrete, (-1, self.dim_input * self.dim_emb))  # (bs, dim_input * dim_emb)
        embeddings = tf.concat((emb_continuous, emb_discrete), axis=-1)  # (bs, dim_input * dim_emb + dim_continuous)

        q = [tower(embeddings, training=training) for tower in self.towers]  # [(bs, dim_hidden_tower), ...]
        for i in range(1, self.num_tasks):
            p = self.gates[i - 1](q[i - 1])  # (bs, dim_hidden_tower)
            q[i] = self.ait(p, q[i])  # (bs, dim_hidden_tower)

        out = tf.concat([head(x) for x, head in zip(q, self.proj_heads)], axis=-1)  # (bs, num_tasks)
        out = tf.nn.sigmoid(out)  # (bs, num_tasks)

        return out
