from ModelUtils.Tester.ModelTester_aghylas import *
from ModelUtils.Models.structurer.ModelName import *
import time as tm

if __name__ == '__main__':
    model_path = '../trained_model/cnn/32/batch_2/'
    model_name = "2_32_cnn_2_1.h5"
    model = load_saved_model(model_path, model_name)

    model_type = CNN
    images_size = 32
    starting_pos = 0

    predict_path = '{}{}\\{}\\'.format(PREDICTIONS_DIR, model_type, images_size, model_name)
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    #Parties à prédire:
    #------------------------------------------------------------------------------------------------------#
    parts = 20#a lancer
    predict_on_test_images(model_type, model, model_name, images_size, starting_pos,images_ids_parts[parts])
    print('==========================>training: {}<======================'.format(parts))
    # for i in [13,14,15,16,7,18,19,20]:
    #     parts = i  # a lancer
    #     predict_on_test_images(model_type, model, model_name, images_size, starting_pos, images_ids_parts[parts])
    #     print('==========================>training: {}<======================'.format(parts))

    #-----------------------------------------------------------------------------------------------------#

    #
    #
    # model_name = "0_16_linear_model_0_4.h5"
    # model = load_saved_model(model_path, model_name)
    # predict_on_test_images(model_type, model, model_name, images_size)
    #
    #
    #
    #
    # model_path = '../trained_model/mlp/16/batch_0/'
    # model_name = "0_16_mlp_0_7.h5"
    # model = load_saved_model(model_path, model_name)
    #
    # model_type = MLP
    # images_size = 16
    # starting_pos = 0
    #
    # predict_on_test_images(model_type, model, model_name, images_size, starting_pos)
    #
    # model_name = "0_16_mlp_0_9.h5"
    # model = load_saved_model(model_path, model_name)
    # predict_on_test_images(model_type, model, model_name, images_size, starting_pos)
    # #
    # #
    # #
    # #
    #
    # model_path = '../trained_model/mlp/32/batch_0/'
    # model_name = "0_32_mlp_0_5.h5"
    # model = load_saved_model(model_path, model_name)
    #
    # model_type = MLP
    # images_size = 32
    #
    # predict_on_test_images(model_type, model, model_name, images_size, starting_pos)
    #
    # model_name = "0_32_mlp_0_7.h5"
    # model = load_saved_model(model_path, model_name)
    # predict_on_test_images(model_type, model, model_name, images_size, starting_pos)
    #
    #
    #
    #
    # model_path = '../trained_model/mlp/64/batch_0/'
    # model_name = "0_64_mlp_0_3.h5"
    # model = load_saved_model(model_path, model_name)
    #
    # model_type = MLP
    # images_size = 64
    #
    # predict_on_test_images(model_type, model, model_name, images_size, starting_pos)
    #
    # model_name = "0_64_mlp_0_4.h5"
    # model = load_saved_model(model_path, model_name)
    # predict_on_test_images(model_type, model, model_name, images_size, starting_pos)