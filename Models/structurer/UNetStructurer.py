from tensorflow.keras.optimizers import *

class UNetStructurer :

    def __init__(self):
        self.name = "unet"
        self.nb_Conv2D_layers = 0
        self.filter = 32
        self.kernel_size = (3, 3)
        self.batch_size = 32
        self.input_shape = (32, 32, 3)
        self.conv2D_activation = 'relu'
        self.output_activation = 'softmax'
        self.use_MaxPooling2D = False
        self.MaxPooling2D_position = []
        self.use_dropout = False
        self.dropout_indexes = []
        self.dropout_value = 0.0
        self.use_l1l2_regularisation_hidden_layers = False
        self.use_l1l2_regularisation_output_layer = False
        self.l1_value = 0.0
        self.l2_value = 0.0
        self.l1l2_regul_indexes = []
        self.loss = 'sparse_categorical_crossentropy'
        self.optimizer = Adam()
        self.metrics = ['accuracy']
        self.padding = 'same'