import tensorflow as tf
from keras import Model, activations
from keras.layers import Dense, Flatten

from models.embedding import MultiInputEmbedding
from models.mlp import MLP


class MixtureOfExperts(Model):
    def __init__(
        self,
        num_tasks: int,
        num_emb: int,
        dim_emb: int = 32,
        embedding_l2: float = 0.0,
        num_experts: int = 1,
        num_hidden_expert: int = 2,
        dim_hidden_expert: int = 64,
        dropout_expert: float = 0.0,
        gate_function: str = "softmax",
        num_hidden_tasks: int = 2,
        dim_hidden_tasks: int = 64,
        dim_out_tasks: int = 1,
        dropout_tasks: float = 0.0,
    ):
        super().__init__()

        # embedding layer
        self.embedding = MultiInputEmbedding(
            num_categorical_emb=num_emb,
            dim_emb=dim_emb,
            regularization=embedding_l2,
        )
        self.flatten = Flatten()

        # experts
        self.experts = []
        for _ in range(num_experts):
            self.experts.append(MLP(num_hidden_expert, dim_hidden_expert, dropout=dropout_expert))

        # encoders for each task
        self.towers = []
        for _ in range(num_tasks):
            self.towers.append(MLP(num_hidden_tasks, dim_hidden_tasks, dim_out_tasks, dropout=dropout_tasks))

        # gate dynamically weights each expert output. A temperature scaling in the softmax might improve performance
        self.gate = Dense(num_experts, activation=gate_function, use_bias=False)

    def call(self, inputs: list[tf.Tensor], training: bool | None = None) -> tf.Tensor:
        sparse_inputs, dense_inputs = inputs

        embeddings = self.embedding([sparse_inputs, dense_inputs])
        embeddings = self.flatten(embeddings)

        out_experts = []
        for expert in self.experts:
            out_experts.append(expert(embeddings, training=training))  # (bs, num_hidden_expert)

        out_experts = tf.stack(out_experts, axis=-1)  # (bs, num_hidden_expert, num_experts)
        gate_score = self.gate(embeddings, training=training)  # (bs, num_experts)
        in_task = tf.einsum("bie,be->bi", out_experts, gate_score)  # (bs, num_hidden_expert)

        out_tasks = []
        for tower in self.towers:
            logits_task = tower(in_task, training=training)  # (bs, 1)
            out_tasks.append(logits_task)

        out = tf.concat(out_tasks, -1)  # (bs, num_tasks)
        out = activations.sigmoid(out)  # (bs, num_tasks)

        return out  # (bs, num_tasks)
