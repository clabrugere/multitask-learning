import tensorflow as tf
from models.mlp import MLP


class MultiGateMixtureOfExperts(tf.keras.Model):
    def __init__(
        self,
        dim_input,
        num_tasks,
        num_emb,
        dim_emb=32,
        embedding_l2=0.0,
        num_experts=1,
        num_hidden_expert=2,
        dim_hidden_expert=64,
        dropout_expert=0.0,
        gate_function="softmax",
        num_hidden_tasks=2,
        dim_hidden_tasks=64,
        dim_out_tasks=1,
        dropout_tasks=0.0,
    ):
        super().__init__()

        self.dim_input = dim_input
        self.num_tasks = num_tasks
        self.dim_emb = dim_emb

        # embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_emb,
            output_dim=dim_emb,
            input_length=dim_input,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=embedding_l2),
        )

        # experts
        self.experts = []
        for _ in range(num_experts):
            self.experts.append(MLP(num_hidden_expert, dim_hidden_expert, dropout=dropout_expert))

        # encoders and gates for each task
        self.towers = []
        self.gates = []
        for _ in range(num_tasks):
            self.towers.append(MLP(num_hidden_tasks, dim_hidden_tasks, dim_out_tasks, dropout=dropout_tasks))
            self.gates.append(tf.keras.layers.Dense(num_experts, activation=gate_function, use_bias=False))

    def call(self, inputs, training=None):
        embeddings = self.embedding(inputs, training=training)  # (bs, dim_input, dim_emb)
        embeddings = tf.reshape(embeddings, (-1, self.dim_input * self.dim_emb))  # (bs, dim_input * dim_emb)

        out_experts = []
        for expert in self.experts:
            out_experts.append(expert(embeddings, training=training))  # (bs, num_hidden_expert)

        out_experts = tf.stack(out_experts, axis=-1)  # (bs, num_hidden_expert, num_experts)

        out_tasks = []
        for gate, tower in zip(self.gates, self.towers):
            gate_score = gate(embeddings, training=training)  # (bs, num_experts)
            in_task = tf.einsum("bie,be->bi", out_experts, gate_score)  # (bs, num_hidden_expert)
            logits_task = tower(in_task, training=training)  # (bs, 1)
            out_tasks.append(logits_task)

        out = tf.concat(out_tasks, -1)  # (bs, num_tasks)
        out = tf.nn.sigmoid(out)  # (bs, num_tasks)

        return out  # (bs, num_tasks)