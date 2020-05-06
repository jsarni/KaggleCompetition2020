from ModelUtils.Models.LinearModel import *
from ModelUtils.Models.MLP import *
from ModelUtils.Models.structurer.LinearStructurer import *
from ModelUtils.Models.structurer.MlpStructurer import MlpStructurer
from ModelUtils.Tester.ModelTester_matthieu import *
from ModelUtils.Models.structurer.ModelName import *
from tensorflow.compat.v1 import set_random_seed
#from tensorflow.keras.backend import clear_session
import tensorflow as tf

if __name__ == '__main__':
    images_size = 16
    my_batch = 1
    nb_train_batches = 1
    nb_test_batches = 1
    validation_split = 0.2
    epochs = 500

    epochs = 2000
    for i in range(3):
        #set_random_seed(10)
        linear_model_struct = LinearStructurer()
        # linear_model_struct.nb_classes = 572
        linear_model_struct.input_shape = (images_size, images_size, 3)
        #model = create_linear_model(linear_model_struct)
        description = getLinStructAsString(linear_model_struct)

        model_path = '{}{}\\{}\\{}\\'.format(TRAINED_MODEL_DIR, "model_lineaire", images_size, "batch_0")
        print(model_path)
        model_name = "0_16_linear_model_0_11.h5"
        model = load_saved_model(model_path, model_name)

        #fit saved model
        model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches, nb_test_batches, validation_split, epochs)
        test_model_batch(LINEAR_MODEL, model, model_name_root, my_batch, images_size, nb_test_batches)
        epochs = epochs * 2

    #model_juba = tf.keras.models.load_model('../trained_model/model_lineaire/16/batch_0/0_16_linear_model_0_11.h5')
    #model_juba.summary()


