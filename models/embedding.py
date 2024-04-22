import tensorflow as tf
from keras import regularizers
from keras.layers import Dense, Embedding, Layer


class MultiInputEmbedding(Layer):
    def __init__(
        self,
        num_categorical_emb: int,
        dim_emb: int,
        bias: bool = False,
        regularization: float = 0.0,
        name: str = "multi_input_embedding",
    ) -> None:
        super().__init__(name=name)

        self.num_sparse_emb = num_categorical_emb
        self.dim_emb = dim_emb
        self.bias = bias
        self.regularization = regularization

    def build(self, inputs_shape):
        _, input_shape_dense = inputs_shape
        dim_dense = input_shape_dense[-1]

        self.sparse_embedding = Embedding(
            self.num_sparse_emb, self.dim_emb, embeddings_regularizer=regularizers.l2(self.regularization)
        )
        self.dense_embedding = Dense(dim_dense * self.dim_emb, use_bias=self.bias)
        self.dim_dense = dim_dense
        self.built = True

    def call(self, inputs):
        sparse_inputs, dense_inputs = inputs

        sparse_outputs = self.sparse_embedding(sparse_inputs)
        dense_outputs = tf.reshape(self.dense_embedding(dense_inputs), (-1, self.dim_dense, self.dim_emb))

        return tf.concat([sparse_outputs, dense_outputs], axis=1)
