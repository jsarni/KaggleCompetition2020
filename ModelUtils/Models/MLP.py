from random import randint, choice
import pandas as pd
from math import isnan

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *

from ModelUtils.Models.structurer.MlpStructurer import MlpStructurer



################################################################## Beginin of MLP Part ##########################################################################################

def create_custom_mlp(mlp_struct: MlpStructurer) -> Model:
    assert mlp_struct.nb_hidden_layers == len(
        mlp_struct.layers_size), "MlpStructurerError: MLP number of layers (nb_hidden_layers) is different of the total layers sizes number (layer_size) "
    assert (mlp_struct.nb_classes > 0)

    model = Sequential()
    model.add(Flatten(input_shape=mlp_struct.input_shape))
    if mlp_struct.use_dropout and 0 in mlp_struct.dropout_indexes:
        model.add(Dropout(mlp_struct.dropout_value, name="dropout_input"))

    for i in range(len(mlp_struct.layers_size)):

        # Hidden layers L1L2 regularisation
        if mlp_struct.use_l1l2_regularisation_hidden_layers and ((i + 1) in mlp_struct.regulization_indexes):
            model.add(Dense(mlp_struct.layers_size[i],
                            activation=mlp_struct.layers_activation,
                            kernel_regularizer=L1L2(l1=mlp_struct.l1_value, l2=mlp_struct.l2_value),
                            name=f"dense_l1l2_{i}"
                            )
                      )
        else:
            model.add(Dense(mlp_struct.layers_size[i],
                            activation=mlp_struct.layers_activation,
                            name=f"dense_{i}"
                            )
                      )
        if mlp_struct.use_dropout and ((i + 1) in mlp_struct.dropout_indexes):
            model.add(Dropout(mlp_struct.dropout_value, name=f"dropout_{i}"))

    # Output L1L2 regularisation
    if mlp_struct.use_l1l2_regularisation_output_layer:
        model.add(Dense(mlp_struct.nb_classes,
                        activation=mlp_struct.output_activation,
                        kernel_regularizer=L1L2(l1=mlp_struct.l1_value, l2=mlp_struct.l2_value),
                        name="output_l1l2"
                        )
                  )
    else:
        model.add(Dense(mlp_struct.nb_classes,
                        activation=mlp_struct.output_activation,
                        name="output"
                        )
                  )

    model.compile(loss=mlp_struct.loss, optimizer=mlp_struct.optimizer, metrics=mlp_struct.metrics)

    return model


def generateMlpModels(nb_hidden_layers_list: list,
                          layers_size_list: list,
                          layers_activation_list: list,
                          output_activation_list: list,
                          use_dropout_list: list,
                          dropout_indexes_list: list,
                          dropout_value_list: list,
                          use_l1l2_regularisation_hidden_layers_list: list,
                          use_l1l2_regularisation_output_layer_list: list,
                          l1_value_list: list,
                          l2_value_list: list,
                          regulization_indexes_list: list,
                          loss_list: list,
                          optimizer_list: list,
                          metrics_list: list):
    assert len(nb_hidden_layers_list) == len(layers_size_list)
    assert len(nb_hidden_layers_list) == len(layers_activation_list)
    assert len(nb_hidden_layers_list) == len(output_activation_list)
    assert len(nb_hidden_layers_list) == len(use_dropout_list)
    assert len(nb_hidden_layers_list) == len(dropout_indexes_list)
    assert len(nb_hidden_layers_list) == len(dropout_value_list)
    assert len(nb_hidden_layers_list) == len(use_l1l2_regularisation_hidden_layers_list)
    assert len(nb_hidden_layers_list) == len(use_l1l2_regularisation_output_layer_list)
    assert len(nb_hidden_layers_list) == len(l1_value_list)
    assert len(nb_hidden_layers_list) == len(l2_value_list)
    assert len(nb_hidden_layers_list) == len(regulization_indexes_list)
    assert len(nb_hidden_layers_list) == len(loss_list)
    assert len(nb_hidden_layers_list) == len(optimizer_list)
    assert len(nb_hidden_layers_list) == len(metrics_list)

    mlp_models = []
    mlp_descriptions = []

    current_structure = MlpStructurer()

    for i in range(len(nb_hidden_layers_list)):
        current_structure.nb_hidden_layers = nb_hidden_layers_list[i]
        current_structure.layers_size = layers_size_list[i]
        current_structure.layers_activation = layers_activation_list[i]
        current_structure.output_activation = output_activation_list[i]
        current_structure.use_dropout = use_dropout_list[i]
        current_structure.dropout_indexes = dropout_indexes_list[i]
        current_structure.dropout_value = dropout_value_list[i]
        current_structure.use_l1l2_regularisation_hidden_layers = use_l1l2_regularisation_hidden_layers_list[i]
        current_structure.use_l1l2_regularisation_output_layer = use_l1l2_regularisation_output_layer_list[i]
        current_structure.l1_value = l1_value_list[i]
        current_structure.l2_value = l2_value_list[i]
        current_structure.regulization_indexes = regulization_indexes_list[i]
        current_structure.loss = loss_list[i]
        current_structure.optimizer = optimizer_list[i]
        current_structure.metrics = metrics_list[i]

        mlp_models.append(create_custom_mlp(current_structure))
        mlp_descriptions.append(getMlpStructAsString(current_structure))

    return mlp_models, mlp_descriptions

def generateRandoMlpStruc(
        nb_classes,
        use_l1l2_hidden=False,
        use_l1l2_output=False,
        use_dropout=False,
        same_layers_depth=True,
        min_nb_layers=5,
        max_nb_layers=40,
        min_layer_depth=32,
        max_layer_depth=512):
    layers_activations = ['softmax', 'relu', 'softplus', 'selu']
    output_activations = ['softmax']
    metrics = [['sparse_categorical_accuracy']]
    losses = ['sparse_categorical_crossentropy']
    optimizers = [Adam()]
    possible_layers_sizes = []
    specific_size = min_layer_depth
    while specific_size <= max_layer_depth:
        possible_layers_sizes.append(specific_size)
        specific_size *= 2
    nb_layers = randint(min_nb_layers, max_nb_layers)
    layers_size = []
    if same_layers_depth:
        depth = choice(possible_layers_sizes)
        layers_size = [depth for i in range(nb_layers)]
    else:
        for i in range(nb_layers):
            layers_size.append(choice(possible_layers_sizes))
    use_dropout = use_dropout
    use_l1l2 = use_l1l2_hidden
    use_l1l2_output = use_l1l2_output
    dropout_indexes = []
    dropout_value = 0.0
    if use_dropout:
        dropout_indexes_number = randint(1, nb_layers)
        dropout_value = randint(1, 3) / 10
        for j in range(dropout_indexes_number):
            dropout_indexes.append(randint(1, nb_layers))
    l1l2_indexes = []
    l1_value = 0.0
    l2_value = 0.0
    if use_l1l2:
        l1l2_indexes_number = randint(1, nb_layers)
        for j in range(l1l2_indexes_number):
            l1l2_indexes.append(randint(1, nb_layers))
        l1_value = randint(5, 100)/1000
        l2_value = randint(5, 100) / 1000

    struct = MlpStructurer()
    struct.nb_classes = nb_classes
    struct.nb_hidden_layers = nb_layers
    struct.layers_size = layers_size
    struct.layers_activation = choice(layers_activations)
    struct.output_activation = choice(output_activations)
    struct.use_dropout = use_dropout
    struct.dropout_indexes = dropout_indexes
    struct.dropout_value = dropout_value
    struct.use_l1l2_regularisation_hidden_layers = use_l1l2
    struct.use_l1l2_regularisation_output_layer = use_l1l2_output
    struct.l1_value = l1_value
    struct.l2_value = l2_value
    struct.regulization_indexes = l1l2_indexes
    struct.loss = choice(losses)
    struct.optimizer = choice(optimizers)
    struct.metrics = choice(metrics)

    return struct

def getMlpStructAsString(mlp_structurer):
    return "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(mlp_structurer.nb_hidden_layers,
                                                                    " ".join([str(i) for i in mlp_structurer.layers_size]),
                                                                    mlp_structurer.layers_activation,
                                                                    mlp_structurer.output_activation,
                                                                    mlp_structurer.use_dropout,
                                                                    str(len(mlp_structurer.dropout_indexes)),
                                                                    " ".join([str(i) for i in mlp_structurer.dropout_indexes]),
                                                                    mlp_structurer.dropout_value,
                                                                    mlp_structurer.use_l1l2_regularisation_hidden_layers,
                                                                    mlp_structurer.use_l1l2_regularisation_output_layer,
                                                                    mlp_structurer.l1_value,
                                                                    mlp_structurer.l2_value,
                                                                    str(len(mlp_structurer.regulization_indexes)),
                                                                    " ".join([str(i) for i in mlp_structurer.regulization_indexes]),
                                                                    mlp_structurer.loss,
                                                                    mlp_structurer.optimizer.__class__.__name__,
                                                                    " ".join(mlp_structurer.metrics)
                                                                    )
def generateStructsFromCSV(file: str):
    structs_descriptions = pd.read_csv(file, sep=';')
    structs_descriptions = structs_descriptions.sort_values('nb_couches', ascending=True)
    structs = []

    for i, row in structs_descriptions.iterrows():
        struct = MlpStructurer()
        struct.nb_hidden_layers = row['nb_couches']
        struct.layers_size = [int(x) for x in row['profondeurs_couches'].split(' ')]
        struct.layers_activation = row['activation_couches']
        struct.output_activation = row['activation_output']
        struct.use_dropout = row['dropout']
        if isnan(row['indexes_dropout']):
            struct.dropout_indexes = []
        else:
            struct.dropout_indexes = [int(x) for x in row['indexes_dropout'].split(' ')]
        struct.dropout_value = row['valeur_dropout']
        struct.use_l1l2_regularisation_hidden_layers = row['l1l2_couches']
        struct.use_l1l2_regularisation_output_layer = row['l1l2_output']
        struct.l1_value = row['valeur_l1']
        struct.l2_value = row['valeur_l2']
        if isnan(row['indexes_l1l2']):
            struct.regulization_indexes = []
        else:
            struct.regulization_indexes = [int(x) for x in row['indexes_l1l2'].split(' ')]
        struct.loss = row['loss']
        struct.optimizer = row['optimizer']
        struct.metrics = [str(x) for x in row['metrics'].split(' ')]

        structs.append(struct)

    return structs

################################################################## End of MLP Part ##########################################################################################
