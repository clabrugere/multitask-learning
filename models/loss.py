import tensorflow as tf
from keras.losses import Loss, binary_crossentropy


class MultiTaskBCE(Loss):
    def __init__(self, num_tasks: int, task_weights: list[float] = None) -> None:
        super().__init__()

        if task_weights is None:
            self.task_weights = tf.ones((1, num_tasks))
        elif tf.rank(task_weights) == 1:
            self.task_weights = tf.expand_dims(task_weights, 0)
        else:
            self.task_weights = task_weights

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        bce = binary_crossentropy(y_true, y_pred)  # (bs, num_tasks)
        loss = self.task_weights * tf.reduce_mean(bce, axis=0)  # (1, num_tasks)
        loss = tf.reduce_sum(loss)  # (1)

        return loss
