from __future__ import absolute_import, division, print_function, unicode_literals

from ModelUtils.Models.structurer.CNNStructurer import *
from tensorflow.keras import layers, models, Model
from tensorflow.keras.regularizers import *



def create_CNN_model(cnn_struct:CNNStructurer,image_size) -> Model:
    assert (cnn_struct.nb_Conv2D_layers == len(cnn_struct.Conv2D_size_layers)), "CNNStructurerError: CNN number of layers  is different of the total layers sizes number "
    assert (cnn_struct.nb_classes > 0)

    inputshape=(image_size,image_size,3)
    model = models.Sequential()
    if cnn_struct.use_l1l2_regularisation_Convolution_layers and (0 in cnn_struct.regul_kernel_indexes):
        model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[0][0]
                                ,(cnn_struct.Conv2D_size_layers[0][1],
                                  cnn_struct.Conv2D_size_layers[0][1]),
                                kernel_regularizer=l1_l2(cnn_struct.l1_value,cnn_struct.l2_value),
                                activation=cnn_struct.Conv2D_activation,
                                input_shape=inputshape,
                                padding=cnn_struct.Conv2D_padding,
                                name="Conv_0_with_l1_l2"
                                )
                  )
    else:
        model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[0][0],
                                (cnn_struct.Conv2D_size_layers[0][1], cnn_struct.Conv2D_size_layers[0][1]),
                                activation=cnn_struct.Conv2D_activation,
                                input_shape=inputshape,
                                padding=cnn_struct.Conv2D_padding,
                                name="Conv_0"
                                )
                  )
    if(cnn_struct.use_dropout and 0 in cnn_struct.dropout_indexes):
        model.add(layers.Dropout(cnn_struct.dropout_value))

    for i in range(1,len(cnn_struct.Conv2D_size_layers)):
        if(cnn_struct.MaxPooling2D_use and i in cnn_struct.MaxPooling2D_Position):

            model.add(layers.MaxPool2D((cnn_struct.MaxPooling2D_values,cnn_struct.MaxPooling2D_values)))
        if  (cnn_struct.use_l1l2_regularisation_Convolution_layers and i+1 in cnn_struct.regul_kernel_indexes):
            model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[i][0],
                                    (cnn_struct.Conv2D_size_layers[i][1],cnn_struct.Conv2D_size_layers[i][1]),
                                    kernel_regularizer=l1_l2(cnn_struct.l1_value, cnn_struct.l2_value),
                                    activation=cnn_struct.Conv2D_activation,
                                    padding=cnn_struct.Conv2D_padding,
                                    name=f"Conv_with_l1_l2_{i}"
                                    )
                      )
        else:
            model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[i][0],
                                    (cnn_struct.Conv2D_size_layers[i][1], cnn_struct.Conv2D_size_layers[i][1]),
                                    activation=cnn_struct.Conv2D_activation,
                                    padding=cnn_struct.Conv2D_padding,
                                    name=f"Conv_{i}"
                                    )
                      )
        if (cnn_struct.use_dropout and i in cnn_struct.dropout_indexes):
                model.add(layers.Dropout(cnn_struct.dropout_value))
    model.add(layers.Flatten())


    if cnn_struct.use_l1l2_regularisation_output_layer:
        model.add(layers.Dense(cnn_struct.nb_classes,
                        activation=cnn_struct.output_activation,
                        kernel_regularizer=L1L2(l1=cnn_struct.l1_value, l2=cnn_struct.l2_value),
                        name="output_l1l2"
                        )
                  )
    else:
        model.add(layers.Dense(cnn_struct.nb_classes,
                        activation=cnn_struct.output_activation,
                        name="output"
                        )
                  )

    model.compile(loss=cnn_struct.loss, optimizer=cnn_struct.optimizer, metrics=cnn_struct.metrics)
    return model



######################################################création de modèles et leurs descriptions  cnn ####################################"

def generateCNNModels(    nb_Conv2D_layers_list: list,
                          Conv2D_layers_size_list: list,
                          Conv2D_activation_list: list,
                          output_activation_list: list,
                          Conv2D_padding_list: list,
                          MaxPooling2D_use_list : list,
                          MaxPooling2D_Position_list :list,
                          MaxPooling2D_values_list:list,
                          use_dropout_list: list,
                          dropout_indexes_list: list,
                          dropout_value_list: list,
                          use_l1l2_regularisation_Conv2D_layers_list: list,
                          use_l1l2_regularisation_output_layer_list: list,
                          l1_value_list: list,
                          l2_value_list: list,
                          regulization_indexes_list: list,
                          loss_list: list,
                          optimizer_list: list,
                          metrics_list: list,
                          image_size:int):
    assert len(nb_Conv2D_layers_list) == len(Conv2D_layers_size_list)
    assert len(nb_Conv2D_layers_list) == len(Conv2D_activation_list)
    assert len(nb_Conv2D_layers_list) == len(output_activation_list)
    assert len(nb_Conv2D_layers_list) == len(Conv2D_padding_list)
    assert len(nb_Conv2D_layers_list) == len(MaxPooling2D_use_list)
    assert len(nb_Conv2D_layers_list) == len(MaxPooling2D_Position_list)
    assert len(nb_Conv2D_layers_list) == len(MaxPooling2D_values_list)
    assert len(nb_Conv2D_layers_list) == len(use_dropout_list)
    assert len(nb_Conv2D_layers_list) == len(dropout_indexes_list)
    assert len(nb_Conv2D_layers_list) == len(dropout_value_list)
    assert len(nb_Conv2D_layers_list) == len(use_l1l2_regularisation_Conv2D_layers_list)
    assert len(nb_Conv2D_layers_list) == len(use_l1l2_regularisation_output_layer_list)
    assert len(nb_Conv2D_layers_list) == len(l1_value_list)
    assert len(nb_Conv2D_layers_list) == len(l2_value_list)
    assert len(nb_Conv2D_layers_list) == len(regulization_indexes_list)
    assert len(nb_Conv2D_layers_list) == len(loss_list)
    assert len(nb_Conv2D_layers_list) == len(optimizer_list)
    assert len(nb_Conv2D_layers_list) == len(metrics_list)

    cnn_models = []
    cnn_descriptions = []

    current_structure = CNNStructurer()

    for i in range(len(nb_Conv2D_layers_list)):
        current_structure.nb_Conv2D_layers   = nb_Conv2D_layers_list[i]
        current_structure.Conv2D_size_layers = Conv2D_layers_size_list[i]
        current_structure.Conv2D_activation  = Conv2D_activation_list[i]
        current_structure.Conv2D_padding=Conv2D_padding_list[i]
        current_structure.output_activation  = output_activation_list[i]
        current_structure.MaxPooling2D_use   = MaxPooling2D_use_list[i]
        current_structure.MaxPooling2D_Position  = MaxPooling2D_Position_list[i]
        current_structure.MaxPooling2D_values  =MaxPooling2D_values_list[i]
        current_structure.use_dropout        = use_dropout_list[i]
        current_structure.dropout_indexes    = dropout_indexes_list[i]
        current_structure.dropout_value      = dropout_value_list[i]
        current_structure.use_l1l2_regularisation_Convolution_layers = use_l1l2_regularisation_Conv2D_layers_list[i]
        current_structure.use_l1l2_regularisation_output_layer = use_l1l2_regularisation_output_layer_list[i]
        current_structure.l1_value = l1_value_list[i]
        current_structure.l2_value = l2_value_list[i]
        current_structure.regul_kernel_indexes = regulization_indexes_list[i]
        current_structure.loss = loss_list[i]
        current_structure.optimizer = optimizer_list[i]
        current_structure.metrics = metrics_list[i]

        cnn_models.append(create_CNN_model(current_structure,image_size))
        cnn_descriptions.append(getcnnStructAsString(current_structure))

    return cnn_models, cnn_descriptions


def getcnnStructAsString(cnn_structurer) -> str:
    return "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(cnn_structurer.nb_Conv2D_layers,
                                                                    " ".join([str(i) for i in cnn_structurer.Conv2D_size_layers]),
                                                                    cnn_structurer.Conv2D_activation,
                                                                    cnn_structurer.output_activation,
                                                                    cnn_structurer.Conv2D_padding,
                                                                    cnn_structurer.MaxPooling2D_use,
                                                                    cnn_structurer.MaxPooling2D_values,
                                                                 " ".join([str(i) for i in cnn_structurer.MaxPooling2D_Position]),
                                                                    cnn_structurer.use_dropout,
                                                                    " ".join([str(i) for i in cnn_structurer.dropout_indexes]),
                                                                    cnn_structurer.dropout_value,
                                                                    cnn_structurer.use_l1l2_regularisation_Convolution_layers,
                                                                    cnn_structurer.use_l1l2_regularisation_output_layer,
                                                                    cnn_structurer.l1_value,
                                                                    cnn_structurer.l2_value,
                                                                    " ".join([str(i) for i in cnn_structurer.regul_kernel_indexes]),
                                                                    cnn_structurer.loss,
                                                                    cnn_structurer.optimizer,
                                                                    " ".join([str(i) for i in cnn_structurer.metrics])
                                                                    )

def nb_maxPooling2D_usedmax(filter:int,kernel:int) -> int:
    res = 0
    if(filter>=kernel):
        res=1
    while (int(filter / kernel) >= kernel):
        res += 1
        filter = int(filter / kernel)
    return res