from ModelUtils.Tester.ModelTester_Juba import *
from ModelUtils.Models.structurer.ModelName import *

if __name__ == '__main__':
    model_path = '../trained_model/model_lineaire/16/batch_0/'
    model_name = "0_16_linear_model_0_2.h5"
    model = load_saved_model(model_path, model_name)

    model_type = LINEAR_MODEL
    images_size = 16


    predict_on_test_images(model_type, model, model_name, images_size)