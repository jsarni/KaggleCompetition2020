from .ModelName import LSTM

class LstmStructurer:

    def __init__(self):
        self.name = LSTM
        self.nb_layers = 3
        self.nb_classes = 0
        self.units = 32
        self.activation = 'tanh'
        self.recurrent_activation = 'sigmoid'
        self.output_activation = 'softmax'
        self.dropout_value = 0.0
        self.recurrent_dropout_value = 0.0
        self.kernel_regularizer = None
        self.recurrent_regularizer = None
        self.output_regularizer = None
        self.l1_value = 0.0
        self.l2_value = 0.0
        self.loss = 'sparse_categorical_crossentropy'
        self.optimizer = 'Adam'
        self.metrics = ['sparse_categorical_accuracy']