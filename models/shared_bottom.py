import tensorflow as tf
from models.mlp import MLP


class SharedBottom(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_tasks,
        dim_continuous,
        num_emb,
        dim_emb=32,
        embedding_l2=0.0,
        num_hidden_shared=2,
        dim_hidden_shared=64,
        dropout_shared=0.0,
        num_hidden_tasks=2,
        dim_hidden_tasks=64,
        dim_out_tasks=1,
        dropout_tasks=0.0,
    ):
        super().__init__()

        self.dim_input = dim_input
        self.num_tasks = num_tasks
        self.dim_continuous = dim_continuous
        self.dim_emb = dim_emb

        self.continuous_proj = tf.keras.layers.Dense(dim_emb)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_emb,
            output_dim=dim_emb,
            input_length=dim_input,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=embedding_l2),
        )

        # shared encoder, it can have any architecture
        self.shared_encoder = MLP(num_hidden_shared, dim_hidden_shared, dropout=dropout_shared)

        # encoders for each task. They can have different architectures depending on the task
        self.towers = []
        for _ in range(num_tasks):
            self.towers.append(MLP(num_hidden_tasks, dim_hidden_tasks, dim_out_tasks, dropout=dropout_tasks))

    def call(self, dense_inputs, discrete_inputs, training=None):
        emb_continuous = self.continuous_proj(dense_inputs, training=training)  # (bs, dim_continuous)
        emb_discrete = self.embedding(discrete_inputs, training=training)  # (bs, dim_input, dim_emb)
        emb_discrete = tf.reshape(emb_discrete, (-1, self.dim_input * self.dim_emb))  # (bs, dim_input * dim_emb)
        embeddings = tf.concat((emb_continuous, emb_discrete), axis=-1)  # (bs, dim_input * dim_emb + dim_continuous)

        latent_shared = self.shared_encoder(embeddings, training=training)  # (bs, dim_hidden_shared)

        out_tasks = []
        for tower in self.towers:
            logits_task = tower(latent_shared, training=training)  # (bs, 1)
            out_tasks.append(logits_task)

        out = tf.concat(out_tasks, -1)  # (bs, num_tasks)
        out = tf.nn.sigmoid(out)  # (bs, num_tasks)

        return out  # (bs, num_tasks)
