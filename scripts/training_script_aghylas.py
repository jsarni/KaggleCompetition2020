from ModelUtils.Models.LinearModel import *
from ModelUtils.Models.MLP import *
from ModelUtils.Models.structurer.LinearStructurer import *
from ModelUtils.Models.structurer.MlpStructurer import MlpStructurer
from ModelUtils.Tester.ModelTester_aghylas import *
from ModelUtils.Models.structurer.ModelName import *
from tensorflow.compat.v1 import set_random_seed
from tensorflow.keras.backend import clear_session

if __name__ == '__main__':
    images_size = 16
    my_batch = 2
    nb_train_batches = 1
    nb_test_batches = 1
    validation_split = 0.2
    epochs = 1000

    batch_size = 4096


    images_size = 32
    nb_train_batches = 1
    nb_test_batches = 1
    validation_split = 0.2
    epochs = 1000

    for i in range(3):
        linear_model_struct = LinearStructurer()
        linear_model_struct.nb_classes = 572
        linear_model_struct.input_shape = (images_size, images_size, 3)
        model = create_linear_model(linear_model_struct)
        description = getLinStructAsString(linear_model_struct)

        # model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, LINEAR_MODEL, images_size)
        # model_name = "16_linear_model_2.h5"
        # model = load_saved_model(model_path, model_name)

        model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches,
                                            nb_test_batches, validation_split, epochs, batch_size=batch_size)
        test_model_batch(LINEAR_MODEL, model, model_name_root, 0, images_size, 1)
        epochs = epochs * 2
        clear_session()

    images_size = 64
    nb_train_batches = 2
    nb_test_batches = 1
    validation_split = 0.2
    epochs = 1000

    for i in range(3):
        linear_model_struct = LinearStructurer()
        linear_model_struct.nb_classes = 572
        linear_model_struct.input_shape = (images_size, images_size, 3)
        model = create_linear_model(linear_model_struct)
        description = getLinStructAsString(linear_model_struct)

        # model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, LINEAR_MODEL, images_size)
        # model_name = "16_linear_model_2.h5"
        # model = load_saved_model(model_path, model_name)

        model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches,
                                            nb_test_batches, validation_split, epochs, batch_size=batch_size)
        test_model_batch(LINEAR_MODEL, model, model_name_root, 0, images_size, 1)
        epochs = epochs * 2
        clear_session()

    images_size = 128
    nb_train_batches = 5
    nb_test_batches = 2
    validation_split = 0.2
    epochs = 1000
    batch_size = 1024
    for i in range(3):
        linear_model_struct = LinearStructurer()
        linear_model_struct.nb_classes = 572
        linear_model_struct.input_shape = (images_size, images_size, 3)
        model = create_linear_model(linear_model_struct)
        description = getLinStructAsString(linear_model_struct)

        # model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, LINEAR_MODEL, images_size)
        # model_name = "16_linear_model_2.h5"
        # model = load_saved_model(model_path, model_name)

        model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches,
                                            nb_test_batches, validation_split, epochs, batch_size=batch_size)
        test_model_batch(LINEAR_MODEL, model, model_name_root, my_batch, images_size, nb_test_batches)
        epochs = epochs * 2
        clear_session()

    images_size = 256
    nb_train_batches = 20
    nb_test_batches = 5
    validation_split = 0.2
    epochs = 1000
    batch_size = 512
    for i in range(3):
        linear_model_struct = LinearStructurer()
        linear_model_struct.nb_classes = 572
        linear_model_struct.input_shape = (images_size, images_size, 3)
        model = create_linear_model(linear_model_struct)
        description = getLinStructAsString(linear_model_struct)

        # model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, LINEAR_MODEL, images_size)
        # model_name = "16_linear_model_2.h5"
        # model = load_saved_model(model_path, model_name)

        model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches,
                                            nb_test_batches, validation_split, epochs, batch_size=batch_size)
        test_model_batch(LINEAR_MODEL, model, model_name_root, my_batch, images_size, nb_test_batches)
        epochs = epochs * 2
        clear_session()

    images_size = 64
    nb_train_batches = 2
    nb_test_batches = 1
    epochs = 500
    batch_size = 2048
    for i in range(3):
        mlp_struct = MlpStructurer()
        mlp_struct.nb_hidden_layers = 5
        mlp_struct.nb_classes = 572
        mlp_struct.input_shape = (images_size, images_size, 3)
        mlp_struct.layers_size = [32, 32, 32, 32, 32]
        model = create_custom_mlp(mlp_struct)
        description = getMlpStructAsString(mlp_struct)
        model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
                                            nb_test_batches, validation_split, epochs, batch_size=batch_size)
        test_model_batch(MLP, model, model_name_root, my_batch, images_size, nb_test_batches)
        epochs = epochs * 2
        clear_session()
    #
    # epochs = 500
    # for i in range(3):
    #     mlp_struct = MlpStructurer()
    #     mlp_struct.nb_hidden_layers = 5
    #     mlp_struct.nb_classes = 572
    #     mlp_struct.input_shape = (images_size, images_size, 3)
    #     mlp_struct.layers_size = [96, 96, 96, 96, 96]
    #     model = create_custom_mlp(mlp_struct)
    #     description = getMlpStructAsString(mlp_struct)
    #     model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs, batch_size=batch_size)
    #     test_model_batch(MLP, model, model_name_root, my_batch, images_size, nb_test_batches)
    #     epochs = epochs * 2
    #     clear_session()

    # for i in range(4):
    #     mlp_struct = MlpStructurer()
    #     mlp_struct.nb_hidden_layers = 5
    #     mlp_struct.nb_classes = 572
    #     mlp_struct.input_shape = (images_size, images_size, 3)
    #     mlp_struct.layers_size = [150, 150, 150, 150, 150]
    #     model = create_custom_mlp(mlp_struct)
    #     description = getMlpStructAsString(mlp_struct)
    #     model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs)
    #     test_model_batch(MLP, model, model_name_root, 0, 16, 1)
    #     epochs = epochs * 2
    #     clear_session()

    # images_size = 32
    # epochs = 500
    # for i in range(3):
    #     mlp_struct = MlpStructurer()
    #     mlp_struct.nb_hidden_layers = 5
    #     mlp_struct.nb_classes = 572
    #     mlp_struct.input_shape = (images_size, images_size, 3)
    #     mlp_struct.layers_size = [32, 32, 32, 32, 32]
    #     model = create_custom_mlp(mlp_struct)
    #     description = getMlpStructAsString(mlp_struct)
    #     model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs)
    #     test_model_batch(MLP, model, model_name_root, 0, images_size, 1)
    #     epochs = epochs * 2
    #     clear_session()
    #
    # epochs = 500
    # for i in range(3):
    #     mlp_struct = MlpStructurer()
    #     mlp_struct.nb_hidden_layers = 5
    #     mlp_struct.nb_classes = 572
    #     mlp_struct.input_shape = (images_size, images_size, 3)
    #     mlp_struct.layers_size = [96, 96, 96, 96, 96]
    #     model = create_custom_mlp(mlp_struct)
    #     description = getMlpStructAsString(mlp_struct)
    #     model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs)
    #     test_model_batch(MLP, model, model_name_root, 0, images_size, 1)
    #     epochs = epochs * 2
    #     clear_session()
    #
    # images_size = 64
    # epochs = 500
    # for i in range(3):
    #     mlp_struct = MlpStructurer()
    #     mlp_struct.nb_hidden_layers = 5
    #     mlp_struct.nb_classes = 572
    #     mlp_struct.input_shape = (images_size, images_size, 3)
    #     mlp_struct.layers_size = [32, 32, 32, 32, 32]
    #     model = create_custom_mlp(mlp_struct)
    #     description = getMlpStructAsString(mlp_struct)
    #     model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs)
    #     test_model_batch(MLP, model, model_name_root, 0, images_size, 1)
    #     epochs = epochs * 2
    #     clear_session()
    #
    # epochs = 500
    # for i in range(3):
    #     mlp_struct = MlpStructurer()
    #     mlp_struct.nb_hidden_layers = 5
    #     mlp_struct.nb_classes = 572
    #     mlp_struct.input_shape = (images_size, images_size, 3)
    #     mlp_struct.layers_size = [96, 96, 96, 96, 96]
    #     model = create_custom_mlp(mlp_struct)
    #     description = getMlpStructAsString(mlp_struct)
    #     model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs)
    #     test_model_batch(MLP, model, model_name_root, 0, images_size, 1)
    #     epochs = epochs * 2
    #     clear_session()
    # for i in range(3):
    #     mlp_struct = MlpStructurer()
    #     mlp_struct.nb_hidden_layers = 5
    #     mlp_struct.nb_classes = 572
    #     mlp_struct.input_shape = (images_size, images_size, 3)
    #     mlp_struct.layers_size = [32, 32, 32, 32, 32]
    #     model = create_custom_mlp(mlp_struct)
    #     description = getMlpStructAsString(mlp_struct)
    #     model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs)
    #     test_model_batch(MLP, model, model_name_root, 0, 16, 1)
    #     epochs = epochs * 2
    #
    # epochs = 500
    # for i in range(3):
    #     mlp_struct = MlpStructurer()
    #     mlp_struct.nb_hidden_layers = 5
    #     mlp_struct.nb_classes = 572
    #     mlp_struct.input_shape = (images_size, images_size, 3)
    #     mlp_struct.layers_size = [96, 96, 96, 96, 96]
    #     model = create_custom_mlp(mlp_struct)
    #     description = getMlpStructAsString(mlp_struct)
    #     model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs)
    #     test_model_batch(MLP, model, model_name_root, 0, 16, 1)
    #     epochs = epochs * 2

    # for i in range(3):
    #     set_random_seed(1)
    #     linear_model_struct = LinearStructurer()
    #     linear_model_struct.nb_classes = 572
    #     linear_model_struct.input_shape = (images_size, images_size, 3)
    #     model = create_linear_model(linear_model_struct)
    #     description = getLinStructAsString(linear_model_struct)
    #
    #     # model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, LINEAR_MODEL, images_size)
    #     # model_name = "16_linear_model_2.h5"
    #     # model = load_saved_model(model_path, model_name)
    #
    #
    #     model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches, nb_test_batches, validation_split, epochs)
    #     test_model_batch(LINEAR_MODEL, model, model_name_root, 0, 16, 1)
    #     epochs = epochs * 2
    #
    # epochs = 2000
    # for i in range(3):
    #     set_random_seed(10)
    #     linear_model_struct = LinearStructurer()
    #     linear_model_struct.nb_classes = 572
    #     linear_model_struct.input_shape = (images_size, images_size, 3)
    #     model = create_linear_model(linear_model_struct)
    #     description = getLinStructAsString(linear_model_struct)
    #
    #     # model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, LINEAR_MODEL, images_size)
    #     # model_name = "16_linear_model_2.h5"
    #     # model = load_saved_model(model_path, model_name)
    #
    #
    #     model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches, nb_test_batches, validation_split, epochs)
    #     test_model_batch(LINEAR_MODEL, model, model_name_root, 0, 16, 1)
    #     epochs = epochs * 2
    #
    # epochs = 2000
    # for i in range(3):
    #     set_random_seed(50)
    #     linear_model_struct = LinearStructurer()
    #     linear_model_struct.nb_classes = 572
    #     linear_model_struct.input_shape = (images_size, images_size, 3)
    #     model = create_linear_model(linear_model_struct)
    #     description = getLinStructAsString(linear_model_struct)
    #
    #     # model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, LINEAR_MODEL, images_size)
    #     # model_name = "16_linear_model_2.h5"
    #     # model = load_saved_model(model_path, model_name)
    #
    #     model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches, nb_test_batches,
    #                       validation_split, epochs)
    #     test_model_batch(LINEAR_MODEL, model, model_name_root, 0, 16, 1)
    #     epochs = epochs * 2
