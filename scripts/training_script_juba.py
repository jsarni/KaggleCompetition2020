from ModelUtils.Models.LinearModel import *
from ModelUtils.Models.structurer.LinearStructurer import *
from ModelUtils.Tester.ModelTester import *
from ModelUtils.Models.structurer.ModelName import *

if __name__ == '__main__':
    images_size = 16
    my_batch = 0
    nb_train_batches = 1
    nb_test_batches = 1
    validation_split = 0.2
    epochs = 1000

    # linear_model_struct = LinearStructurer()
    # linear_model_struct.nb_classes = 572
    # linear_model_struct.input_shape = (images_size, images_size, 3)
    # model = create_linear_model(linear_model_struct)
    # description = getLinStructAsString(linear_model_struct)

    model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, LINEAR_MODEL, images_size)
    model_name = "16_linear_model_2.h5"
    model_name_root = "16_linear_model_2"
    model = load_saved_model(model_path, model_name)


    # train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches, nb_test_batches, validation_split, epochs)
    test_model_batch(LINEAR_MODEL, model, model_name_root, 0, 16, 1)
