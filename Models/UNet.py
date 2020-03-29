from random import randint, choice

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *

from Models.structurer.UNetStructurer import UNetStructurer

################################################################## Begining of UNet Part ##########################################################################################

def create_unet(unet_struct: UNetStructurer) -> Model:

    input_tensor = Input((32, 32, 3))

    layers_list = []
    tensors_to_connect_list_1 = []

    for i in range(unet_struct.nb_Conv2D_layers):
        if unet_struct.use_l1l2_regularisation_hidden_layers and ((i + 1) in unet_struct.l1l2_regul_indexes):
            layer = Conv2D(filters=unet_struct.filter,
                           kernel_size=unet_struct.kernel_size,
                           activation=unet_struct.conv2D_activation,
                           kernel_regularizer=L1L2(unet_struct.l1_value, unet_struct.l2_value),
                           name=f"conv2d_l1l2_{i}",
                           padding=unet_struct.padding)
        else:
            layer = Conv2D(filters=unet_struct.filter,
                           kernel_size=unet_struct.kernel_size,
                           activation=unet_struct.conv2D_activation,
                           name=f"conv2d_{i}",
                           padding=unet_struct.padding)

        layers_list.append(layer)

    if unet_struct.use_dropout and (0 in unet_struct.dropout_indexes):
        layers_list[0] = layers_list[0](Dropout(unet_struct.dropout_value, name="dropout_input")(input_tensor))
    else:
        layers_list[0] = layers_list[0](input_tensor)

    if (unet_struct.nb_Conv2D_layers % 2 == 0):
        middle = int(unet_struct.nb_Conv2D_layers / 2)
    else :
        middle = int((unet_struct.nb_Conv2D_layers / 2) + 1)

    upsambled_layers_indexes = [(unet_struct.nb_Conv2D_layers - x) for x in unet_struct.MaxPooling2D_position if x <= middle]

    for j in range(middle - 1):
        tensors_to_connect_list_1.append(layers_list[j])
        if unet_struct.use_MaxPooling2D and (j+1 in unet_struct.MaxPooling2D_position):
            layers_list[j] = MaxPool2D(pool_size=(2, 2), name=f"maxpool_{j}")(layers_list[j])
        if unet_struct.use_dropout and (j+1 in unet_struct.dropout_indexes):
            layers_list[j] = Dropout(unet_struct.dropout_value, name=f"dropout_{j}")(layers_list[j])
        layers_list[j+1] = layers_list[j+1](layers_list[j])

    for j in range(middle, unet_struct.nb_Conv2D_layers):
        if j in upsambled_layers_indexes:
            layers_list[j - 1] = UpSampling2D(name=f"upsample_{j}")(layers_list[j - 1])
        tensors_to_connect_2 = layers_list[j-1]
        tensors_to_connect_1 = tensors_to_connect_list_1.pop()
        avg_tensor = Average()([tensors_to_connect_2, tensors_to_connect_1])
        layers_list[j] = layers_list[j](avg_tensor)

    flatten_tensor = Flatten(name="flatten")(layers_list[-1])
    output_tensor = Dense(10, activation=unet_struct.output_activation, name="output_dense")(flatten_tensor)

    model = Model(input_tensor, output_tensor)

    model.compile(loss=unet_struct.loss, optimizer=unet_struct.optimizer, metrics=unet_struct.metrics)

    return model


def getUNetStructAsString(unet_structurer: UNetStructurer) -> str:
    return "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(unet_structurer.nb_Conv2D_layers,
                                                                                unet_structurer.filter,
                                                                                " ".join([str(i) for i in list(unet_structurer.kernel_size)]),
                                                                                unet_structurer.batch_size,
                                                                                " ".join([str(i) for i in list(unet_structurer.input_shape)]),
                                                                                unet_structurer.conv2D_activation,
                                                                                unet_structurer.output_activation,
                                                                                unet_structurer.use_MaxPooling2D,
                                                                                " ".join([str(i) for i in unet_structurer.MaxPooling2D_position]),
                                                                                unet_structurer.use_dropout,
                                                                                " ".join([str(i) for i in unet_structurer.dropout_indexes]),
                                                                                unet_structurer.dropout_value,
                                                                                unet_structurer.use_l1l2_regularisation_hidden_layers,
                                                                                unet_structurer.use_l1l2_regularisation_output_layer,
                                                                                unet_structurer.l1_value,
                                                                                unet_structurer.l2_value,
                                                                                " ".join([str(i) for i in unet_structurer.l1l2_regul_indexes]),
                                                                                unet_structurer.loss,
                                                                                unet_structurer.optimizer.__class__.__name__,
                                                                                " ".join([i for i in unet_structurer.metrics]),
                                                                                unet_structurer.padding)

def generateRandoUNetStruc(
        use_maxpool=False,
        use_l1l2_hidden=False,
        use_l1l2_output=False,
        use_dropout=False,
        min_nb_layers=3,
        max_nb_layers=20) -> UNetStructurer:
    layers_activations = ['selu']
    output_activations = ['softmax']
    kernel_sizes = [(3, 3)]
    filters = [32]
    batch_sizes = [32]
    metrics = [['sparse_categorical_accuracy']]
    losses = ['sparse_categorical_crossentropy']
    optimizers = [Adam()]
    nb_layers = randint(min_nb_layers, max_nb_layers)
    use_dropout = use_dropout
    use_l1l2 = use_l1l2_hidden
    use_l1l2_output = use_l1l2_output
    dropout_indexes = []
    dropout_value = 0.0
    if use_dropout:
        dropout_indexes_number = randint(1, nb_layers)
        dropout_value = randint(0, 4) / 10
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

    maxpool_indexes = []

    if use_maxpool:
        nb_maxpool_layers = randint(1, int(nb_layers/2))
        for j in range(nb_maxpool_layers):
            maxpool_indexes.append(randint(1, int(nb_layers/2)))

    struct = UNetStructurer()

    struct.nb_Conv2D_layers = nb_layers
    struct.filter = choice(filters)
    struct.kernel_size = choice(kernel_sizes)
    struct.batch_size = choice(batch_sizes)
    struct.input_shape = (32, 32, 3)
    struct.conv2D_activation = choice(layers_activations)
    struct.output_activation = choice(output_activations)
    struct.use_MaxPooling2D = use_maxpool
    struct.MaxPooling2D_position = maxpool_indexes
    struct.use_dropout = use_dropout
    struct.dropout_indexes = dropout_indexes
    struct.dropout_value = dropout_value
    struct.use_l1l2_regularisation_hidden_layers = use_l1l2_hidden
    struct.use_l1l2_regularisation_output_layer = use_l1l2_output
    struct.l1_value = l1_value
    struct.l2_value = l2_value
    struct.l1l2_regul_indexes = l1l2_indexes
    struct.loss = choice(losses)
    struct.optimizer = choice(optimizers)
    struct.metrics = choice(metrics)
    struct.padding = 'same'

    return struct


################################################################## End of UNet Part ##########################################################################################
