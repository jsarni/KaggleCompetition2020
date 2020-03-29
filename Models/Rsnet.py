from random import randint, choice
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *

from Models.structurer.RsnetStructurer import RsnetStructurer


def create_model_resenet34(RsnetStruct: RsnetStructurer) -> Model:


    input_tensor = Input(RsnetStruct.input_shape)
    last_output_tensor = input_tensor

    antipen_output_tensor = None
    nb_skipped = 0
    for i in range(RsnetStruct.nb_hidden_layers):
        if RsnetStruct.use_skip and antipen_output_tensor is not None:
            if nb_skipped == RsnetStruct.nb_skip:
                if (RsnetStruct.use_MaxPooling2D ):
                    k=i
                    for j in range (2):
                        if ( k in RsnetStruct.MaxPooling2D_position):
                            last_output_tensor=UpSampling2D(name=f"upsample_{k}")(last_output_tensor)

                        k -=1
                    add_tensor = Add()([antipen_output_tensor, last_output_tensor])
                else:
                    add_tensor = Add()([antipen_output_tensor, last_output_tensor])

                antipen_output_tensor = add_tensor
                last_output_tensor = add_tensor
                nb_skipped = 0

                # Hidden layers L1L2 regularisation
            if RsnetStruct.use_l1l2_regularisation_hidden_layers and ((i + 1) in RsnetStruct.regulization_indexes):
                last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding, activation=RsnetStruct.layers_activation,
                                            kernel_regularizer=L1L2(l1=RsnetStruct.l1_value, l2=RsnetStruct.l2_value), input_shape=(32, 32, 3),name=f"conv2d__L1L2_{i}")(last_output_tensor)
            else:
                last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding,
                                                activation=RsnetStruct.layers_activation, input_shape=(32, 32, 3),name=f"conv2d_{i}")(last_output_tensor)
            # Use dropout
            if (RsnetStruct.use_dropout and (i + 1) in RsnetStruct.dropout_indexes):
                    last_output_tensor = Dropout(RsnetStruct.dropout_value, name=f"dropout_{i}")(last_output_tensor)

            if (RsnetStruct.use_MaxPooling2D and  (i + 1) in RsnetStruct.MaxPooling2D_position):
                last_output_tensor = MaxPool2D(RsnetStruct.nb_skip, name=f"Maxpool2D_{i}")(last_output_tensor)

            nb_skipped += 1

        else:
            if( RsnetStruct.use_skip and i==0):
                if (RsnetStruct.use_dropout):
                    last_output_tensor = Dropout(RsnetStruct.dropout_value, name=f"dropout_input")(last_output_tensor)

                # Hidden layers L1L2 regularisation
                if RsnetStruct.use_l1l2_regularisation_hidden_layers and ((i + 1) in RsnetStruct.regulization_indexes):
                    last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding,activation=RsnetStruct.layers_activation,
                                            kernel_regularizer=L1L2(l1=RsnetStruct.l1_value,  l2=RsnetStruct.l2_value),input_shape=(32, 32, 3),  name=f"conv2d__L1L2_{i}")(last_output_tensor)
                else:
                    last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding,
                                                activation=RsnetStruct.layers_activation, input_shape=(32, 32, 3),name=f"conv2d_{i}")(last_output_tensor)
                # Use dropout
                if(RsnetStruct.use_dropout and (i+1) in RsnetStruct.dropout_indexes):
                    last_output_tensor = Dropout(RsnetStruct.dropout_value, name=f"dropout_{i}")(last_output_tensor)
                # Use Maxpool2D
                if (RsnetStruct.use_MaxPooling2D and (i + 1) in RsnetStruct.MaxPooling2D_position):
                    last_output_tensor = MaxPool2D(RsnetStruct.nb_skip, name=f"Maxpool2D_{i}")(last_output_tensor)

                if(RsnetStruct.use_dropout):
                    antipen_output_tensor = Dropout(RsnetStruct.dropout_value, name=f"dropout_input_dense")(
                        input_tensor)
                    antipen_output_tensor = Dense(RsnetStruct.filters, activation=RsnetStruct.layers_activation,
                                                  name=f"dense_{i}")(antipen_output_tensor)
                else:
                    antipen_output_tensor = Dense(RsnetStruct.filters, activation=RsnetStruct.layers_activation,
                                                  name=f"dense_{i}")(input_tensor)


            else:
                antipen_output_tensor = input_tensor
                # Hidden layers L1L2 regularisation
                if RsnetStruct.use_l1l2_regularisation_hidden_layers and ((i + 1) in RsnetStruct.regulization_indexes):
                    last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding,activation=RsnetStruct.layers_activation,
                                            kernel_regularizer=L1L2(l1=RsnetStruct.l1_value,l2=RsnetStruct.l2_value),input_shape=(32, 32, 3),name=f"conv2d__L1L2_{i}")(last_output_tensor)
                else:
                    last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding,
                                                activation=RsnetStruct.layers_activation, input_shape=(32, 32, 3), name=f"conv2d_{i}")(last_output_tensor)
                # Use dropout
                if (RsnetStruct.use_dropout and (i + 1) in RsnetStruct.dropout_indexes):
                    last_output_tensor = Dropout(RsnetStruct.dropout_value, name=f"dropout_{i}")(last_output_tensor)

                # Use Maxpool2D
                if (RsnetStruct.use_MaxPooling2D and (i + 1) in RsnetStruct.MaxPooling2D_position):
                    last_output_tensor = MaxPool2D(RsnetStruct.nb_skip, name=f"Maxpool2D_{i}")(last_output_tensor)
                if (i == 0 or i == 1):
                    if (RsnetStruct.use_MaxPooling2D):
                        k = i
                        for j in range(2):
                            if (k + 1 in RsnetStruct.MaxPooling2D_position):
                                last_output_tensor = UpSampling2D(name=f"upsample_{k}")(last_output_tensor)

                            k -= 1
            nb_skipped += 1

    if RsnetStruct.use_skip and RsnetStruct.nb_hidden_layers % RsnetStruct.nb_skip == 0:
        last_output_tensor = Add()([antipen_output_tensor, last_output_tensor])


    flattended_last_tensor = Flatten(name='flatten')(last_output_tensor)
    if (RsnetStruct.use_l1l2_regularisation_output_layer):
        output_tensor = Dense(10,RsnetStruct.output_activation, kernel_regularizer=L1L2(l1=RsnetStruct.l1_value,l2=RsnetStruct.l2_value),name='dense_output_L1L2')(flattended_last_tensor)
    else:
        output_tensor = Dense(10, RsnetStruct.output_activation, name='dense_output')(flattended_last_tensor)
    model = Model(input_tensor, output_tensor)

    model.compile(loss=RsnetStruct.loss,
                  optimizer=RsnetStruct.optimizer,
                  metrics=RsnetStruct.metrics)

    return model



def getResetStructAsString(rsnet_structurer: RsnetStructurer) -> str:
    return "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(rsnet_structurer.nb_hidden_layers,
                                                                                rsnet_structurer.filters,
                                                                                " ".join([str(i) for i in list(rsnet_structurer.kernel_size)]),
                                                                                rsnet_structurer.batch_size,
                                                                                " ".join([str(i) for i in list(rsnet_structurer.input_shape)]),
                                                                                rsnet_structurer.layers_activation,
                                                                                rsnet_structurer.output_activation,
                                                                                rsnet_structurer.use_skip,
                                                                                rsnet_structurer.nb_skip,
                                                                                rsnet_structurer.use_dropout,
                                                                                " ".join([str(i) for i in rsnet_structurer.dropout_indexes]),
                                                                                rsnet_structurer.dropout_value,
                                                                                rsnet_structurer.use_l1l2_regularisation_hidden_layers,
                                                                                rsnet_structurer.use_l1l2_regularisation_output_layer,
                                                                                rsnet_structurer.l1_value,
                                                                                rsnet_structurer.l2_value,
                                                                                " ".join([str(i) for i in rsnet_structurer.regulization_indexes]),
                                                                                rsnet_structurer.loss,
                                                                                rsnet_structurer.optimizer.__class__.__name__,
                                                                                " ".join([i for i in rsnet_structurer.metrics]),
                                                                                rsnet_structurer.padding)

def generateRandomRsnetStruc(
        use_maxpool=False,
        use_l1l2_hidden=False,
        use_l1l2_output=False,
        use_dropout=False,
        use_skip = True,
        nb_skip =2,
        min_nb_layers=3,
        max_nb_layers=10,
        filters=32) -> RsnetStructurer:
    layers_activations = ['softmax', 'relu', 'softplus', 'selu']
    output_activations = ['softmax']
    kernel_sizes = [(3, 3)]
    filters = [32,64]
    batch_sizes = [128]
    metrics = [['categorical_accuracy']]
    losses = ['categorical_crossentropy']
    optimizers = [Adam()]
    nb_hidden_layers =randint(min_nb_layers, max_nb_layers)
    use_dropout = use_dropout
    use_l1l2 = use_l1l2_hidden
    use_l1l2_output = use_l1l2_output
    dropout_indexes = []
    dropout_value = 0.024
    l1_value =0.0
    l2_value = 0.0
    l1l2_indexes =[]
    maxpool_indexes =[]
    if use_dropout:
        dropout_indexes_number = randint(1, nb_hidden_layers) // 2
        dropout_value = randint(0, 4) / 12
        for j in range(dropout_indexes_number):
            dropout_indexes.append(randint(1, nb_hidden_layers))


    if use_l1l2:
        l1l2_indexes_number = randint(1, nb_hidden_layers) // 2
        for j in range(l1l2_indexes_number):
            l1l2_indexes.append(randint(1, nb_hidden_layers))
        l1_value = randint(5, 40) / 1000
        l2_value = randint(5, 40) / 1000



    if use_maxpool:
        nb_maxpool_layers = randint(1, nb_hidden_layers) //2
        for j in range(nb_maxpool_layers):
            maxpool_indexes.append(randint(1, nb_hidden_layers))


    struct = RsnetStructurer()



    struct.nb_hidden_layers = nb_hidden_layers
    struct.filters = choice(filters)
    struct.kernel_size = choice(kernel_sizes)
    struct.batch_size = choice(batch_sizes)
    struct.use_skip=use_skip
    struct.nb_skip=nb_skip
    struct.input_shape = (32, 32, 3)
    struct.layers_activation = choice(layers_activations)
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
    struct.regulization_indexes = l1l2_indexes
    struct.loss = choice(losses)
    struct.optimizer = choice(optimizers)
    struct.metrics = choice(metrics)
    struct.padding = 'same'

    return struct
