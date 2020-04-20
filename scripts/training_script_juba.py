from ModelUtils.Models.LinearModel import *
from ModelUtils.Models.structurer.LinearStructurer import *
from ModelUtils.Tester.ModelTester import train_model
from ModelUtils.Models.structurer.ModelName import *

if __name__ == '__main__':
    images_size = 256
    my_batch = 0
    nb_train_batches = 1
    nb_test_batches = 1
    validation_split = 0.2
    epochs = 10

    linear_model_struct = LinearStructurer()
    linear_model_struct.nb_classes = 572
    linear_model_struct.input_shape = (images_size, images_size, 3)

    model = create_linear_model(linear_model_struct)
    description = getLinStructAsString(linear_model_struct)
    train_model(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches, nb_test_batches, validation_split, epochs)