import tensorflow as tf

from models.mlp import MLP


class SharedBottom(tf.keras.Model):
    def __init__(
        self,
        num_tasks: int,
        dim_categorical: int,
        num_emb: int,
        dim_emb: int = 32,
        embedding_l2: float = 0.0,
        num_hidden_shared: int = 2,
        dim_hidden_shared: int = 64,
        dropout_shared: float = 0.0,
        num_hidden_tasks: int = 2,
        dim_hidden_tasks: int = 64,
        dim_out_tasks: int = 1,
        dropout_tasks: int = 0.0,
    ) -> None:
        super().__init__()

        self.dim_categorical = dim_categorical
        self.num_tasks = num_tasks
        self.dim_emb = dim_emb

        self.continuous_proj = tf.keras.layers.Dense(dim_emb)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_emb,
            output_dim=dim_emb,
            input_length=dim_categorical,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=embedding_l2),
        )

        # shared encoder, it can have any architecture
        self.shared_encoder = MLP(num_hidden_shared, dim_hidden_shared, dropout=dropout_shared)

        # encoders for each task. They can have different architectures depending on the task
        self.towers = []
        for _ in range(num_tasks):
            self.towers.append(MLP(num_hidden_tasks, dim_hidden_tasks, dim_out_tasks, dropout=dropout_tasks))

    def call(self, inputs: list[tf.Tensor], training: bool | None = None) -> tf.Tensor:
        dense_inputs, categorical_inputs = inputs

        emb_continuous = self.continuous_proj(dense_inputs, training=training)
        emb_categorical = self.embedding(categorical_inputs, training=training)
        emb_categorical = tf.reshape(emb_categorical, (-1, self.dim_categorical * self.dim_emb))

        embeddings = tf.concat((emb_continuous, emb_categorical), axis=-1)

        latent_shared = self.shared_encoder(embeddings, training=training)

        out_tasks = []
        for tower in self.towers:
            logits_task = tower(latent_shared, training=training)
            out_tasks.append(logits_task)

        out = tf.concat(out_tasks, -1)
        out = tf.nn.sigmoid(out)

        return out
