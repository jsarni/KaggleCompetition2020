from tensorflow.keras.optimizers import *

class RsnetStructurer:

    def __init__(self):
        self.name = "resnet34"
        self.kernel_size = (3, 3)
        self.filters = 64
        self.batch_size = 128
        self.input_shape = (32, 32, 3)
        self.nb_hidden_layers = 7
        self.layers_activation = 'relu'
        self.output_activation = 'softmax'
        self.padding = "same"
        self.nb_skip = 2
        self.use_skip = True
        self.use_dropout = False
        self.dropout_indexes = []
        self.dropout_value = 0.0
        self.use_MaxPooling2D =False
        self.MaxPooling2D_position =[7]
        self.use_l1l2_regularisation_hidden_layers = False
        self.use_l1l2_regularisation_output_layer = False
        self.l1_value = 0.0
        self.l2_value = 0.0
        self.regulization_indexes = []
        self.loss = 'categorical_crossentropy'
        self.optimizer = Adam()
        self.metrics = ['categorical_accuracy']





