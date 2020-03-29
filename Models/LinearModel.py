from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import Model

from Models.structurer.LinearStructurer import LinearStructurer


def create_lin_model(lin_struct: LinearStructurer) -> Model:
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(10, activation=lin_struct.output_activation))
    model.compile(loss=lin_struct.loss, metrics=lin_struct.metrics)
    return model


def getLinStructAsString(lin_struct: LinearStructurer) -> str:
    return "{};{};{};{};{};{};{}".format(lin_struct.nb_hidden_layers,
                                         " ".join([str(i) for i in lin_struct.layers_size]),
                                         lin_struct.use_layers_activation,
                                         lin_struct.layers_activation,
                                         lin_struct.output_activation,
                                         lin_struct.loss,
                                         lin_struct.optimizer.__class__.__name__,
                                         " ".join(lin_struct.metrics)
                                         )