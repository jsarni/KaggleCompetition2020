from .ModelName import MLP

class MlpStructurer:

    def __init__(self):
        self.name = MLP
        self.nb_hidden_layers = 0
        self.nb_classes = 0
        self.layers_size = []
        self.layers_activation = 'relu'
        self.output_activation = 'softmax'
        self.use_dropout = False
        self.dropout_indexes = []
        self.dropout_value = 0.0
        self.use_l1l2_regularisation_hidden_layers = False
        self.use_l1l2_regularisation_output_layer = False
        self.l1_value = 0.0
        self.l2_value = 0.0
        self.regulization_indexes = []
        self.loss = 'sparse_categorical_crossentropy'
        self.optimizer = 'Adam'
        self.metrics = ['sparse_categorical_accuracy']