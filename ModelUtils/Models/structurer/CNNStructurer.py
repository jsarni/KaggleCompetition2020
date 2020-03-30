from __future__ import absolute_import, division, print_function, unicode_literals
from .ModelName import CNN

class CNNStructurer():
    def __init__(self):
        self.name = CNN  # nom de la structure
        self.nb_classes = 0
        self.nb_Conv2D_layers = 3  # nombre de couches cachées
        self.Conv2D_size_layers = [(32, 3), (32, 3), (64, 3)]  # [input,filter_dimension] dans l'appel on utilisera un couple ( filter_dimension,filter_dimension)
        self.Conv2D_activation = 'relu'
        self.Conv2D_padding='same'
        self.MaxPooling2D_use = True
        self.MaxPooling2D_Position = [2, 3]  # Positionnement des couches Max2Pooling
        self.MaxPooling2D_values = 3  # valeur du filtre Max2Pooling
        self.output_activation = 'softmax'  # activation output
        self.use_dropout = False
        self.dropout_indexes = []
        self.dropout_value = 0
        self.use_l1l2_regularisation_Convolution_layers = False
        self.use_l1l2_regularisation_output_layer = False
        self.l1_value = 0.00
        self.l2_value = 0.00
        self.regul_kernel_indexes = []
        self.loss = 'sparse_categorical_crossentropy'
        self.optimizer = 'Adam'
        self.metrics = ['accuracy']

