import tensorflow as tf
from keras import Model, activations
from keras.layers import Flatten

from models.embedding import MultiInputEmbedding
from models.mlp import MLP


class SharedBottom(Model):
    def __init__(
        self,
        num_tasks: int,
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

        self.embedding = MultiInputEmbedding(
            num_categorical_emb=num_emb,
            dim_emb=dim_emb,
            regularization=embedding_l2,
        )
        self.flatten = Flatten()

        # shared encoder, it can have any architecture
        self.shared_encoder = MLP(num_hidden_shared, dim_hidden_shared, dropout=dropout_shared)

        # encoders for each task. They can have different architectures depending on the task
        self.towers = []
        for _ in range(num_tasks):
            self.towers.append(MLP(num_hidden_tasks, dim_hidden_tasks, dim_out_tasks, dropout=dropout_tasks))

    def call(self, inputs: list[tf.Tensor], training: bool | None = None) -> tf.Tensor:
        sparse_inputs, dense_inputs = inputs

        embeddings = self.embedding([sparse_inputs, dense_inputs])
        embeddings = self.flatten(embeddings)
        latent_shared = self.shared_encoder(embeddings, training=training)

        out_tasks = []
        for tower in self.towers:
            logits_task = tower(latent_shared, training=training)
            out_tasks.append(logits_task)

        out = tf.concat(out_tasks, -1)
        out = activations.sigmoid(out)

        return out
