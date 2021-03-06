from ModelUtils.Models.LinearModel import *
from ModelUtils.Models.MLP import *
from ModelUtils.Models.Rsnet import *
from ModelUtils.Models.structurer.RsnetStructurer import *
from ModelUtils.Models.structurer.LinearStructurer import *
from ModelUtils.Models.structurer.MlpStructurer import MlpStructurer
from ModelUtils.Tester.ModelTester_merzouk import *
from ModelUtils.Models.structurer.ModelName import *
from tensorflow.compat.v1 import set_random_seed
from tensorflow.keras.backend import clear_session

if __name__ == '__main__':
    images_size = 32
    my_batch = 3
    nb_train_batches = 1
    nb_test_batches = 1
    validation_split = 0.2
    epochs = 500

    model_path = '../trained_model/mlp/32/batch_0/'
    model_name = "0_32_mlp_0_5.h5"
    model = load_saved_model(model_path, model_name)

    description= '0_32_mlp_0_5;5 layers; 96 par couche; epochs=500; sparse_categorical_crossentropy;adam;sparse_categorical_accuracy; validation_split=0.2;seed=default'
    model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches, nb_test_batches, validation_split, epochs)
    test_model_batch(MLP, model, model_name_root, 3, images_size, 1)
    clear_session()




    # resnet_struct = RsnetStructurer()
    # resnet_struct.input_shape =(images_size,images_size,3)
    # resnet_struct.loss='sparse_categorical_crossentropy'
    # resnet_struct.metrics = ['categorical_accuracy']
    # model = create_model_resenet34(resnet_struct)
    # description = getResetStructAsString(resnet_struct)
    # model_name_root = train_model_batch(RESNET34, model, description, my_batch, images_size, nb_train_batches, nb_test_batches, validation_split, epochs,batch_size=1200)
    # test_model_batch(RESNET34, model, model_name_root, 3, images_size, 1)
    # clear_session()




    #
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
    #     test_model_batch(MLP, model, model_name_root, 3, images_size, 1)
    #     epochs = epochs * 2
    #     clear_session()
    #
    # images_size = 128
    # nb_train_batches = 5
    # nb_test_batches = 2
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
    #     test_model_batch(MLP, model, model_name_root, 3, images_size, 1)
    #     epochs = epochs * 2
    #     clear_session()
    #
    # images_size = 256
    # epochs = 500
    # for i in range(3):
    #     mlp_struct = MlpStructurer()
    #     mlp_struct.nb_hidden_layers = 5
    #     mlp_struct.nb_classes = 572
    #     mlp_struct.input_shape = (images_size, images_size, 3)
    #     mlp_struct.layers_size = [64, 64, 64, 64, 64]
    #     model = create_custom_mlp(mlp_struct)
    #     description = getMlpStructAsString(mlp_struct)
    #     model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs)
    #     test_model_batch(MLP, model, model_name_root, 3, images_size, 1)
    #     epochs = epochs * 2
    #     clear_session()
    #
    #
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
    #     test_model_batch(MLP, model, model_name_root, 3, images_size, 1)
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
    #     test_model_batch(MLP, model, model_name_root, 3, images_size, 1)
    #     epochs = epochs * 2
    #     clear_session()
    # # for i in range(3):
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
