from keras import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, ReLU


class MLP(Sequential):
    def __init__(
        self,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int | None = None,
        dropout: float = 0.0,
        name: str | None = "MLP",
    ):
        layers = []
        for _ in range(num_hidden - 1):
            layers.append(Dense(dim_hidden))
            layers.append(BatchNormalization())
            layers.append(ReLU())

            if dropout > 0.0:
                layers.append(Dropout(dropout))

        if dim_out:
            layers.append(Dense(dim_out))
        else:
            layers.append(Dense(dim_hidden))

        super().__init__(layers, name=name)
