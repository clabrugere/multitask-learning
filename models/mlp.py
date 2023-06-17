import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self, num_hidden, dim_hidden, dim_out=None, dropout=0.0, batch_norm=True, name="MLP"):
        super().__init__(name=name)

        self.dim_out = dim_out
        self.blocks = tf.keras.Sequential(name="MLP")
        for _ in range(num_hidden - 1):
            self.blocks.add(tf.keras.layers.Dense(dim_hidden))

            if batch_norm:
                self.blocks.add(tf.keras.layers.BatchNormalization())

            self.blocks.add(tf.keras.layers.ReLU())
            self.blocks.add(tf.keras.layers.Dropout(dropout))

        if dim_out:
            self.blocks.add(tf.keras.layers.Dense(dim_out))
        else:
            self.blocks.add(tf.keras.layers.Dense(dim_hidden))

    def call(self, inputs, training=False):
        out = self.blocks(inputs, training=training)

        return out
