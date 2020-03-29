class LinearStructurer:

    def __init__(self):
        self.name = "lin"
        self.nb_hidden_layers = 0
        self.layers_size = []
        self.use_layers_activation = False
        self.layers_activation = 'relu'
        self.output_activation = 'softmax'
        self.loss = 'sparse_categorical_crossentropy'
        self.optimizer = 'Adam'
        self.metrics = ['sparse_categorical_accuracy']