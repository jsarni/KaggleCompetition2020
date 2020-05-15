from ModelUtils.Models.LinearModel import *
from ModelUtils.Models.MLP import *
from ModelUtils.Models.CNN import *
from ModelUtils.Models.Rsnet import *
from ModelUtils.Models.structurer.LinearStructurer import *
from ModelUtils.Models.structurer.CNNStructurer import *
from ModelUtils.Models.structurer.RsnetStructurer import *
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
    epochs = 2000
    batch_size = 4096
    set_random_seed(50)

    #executer un predict:
    # model_path = '../trained_model/model_lineaire/16/batch_2/'
    # model_name = "0_1_2_16_linear_model_2_8.h5"
    # model = load_saved_model(model_path, model_name)
    # model_type = LINEAR_MODEL
    # images_size = 16
    # starting_pos = 0
    # predict_on_test_images(model_type, model, model_name, images_size, starting_pos)
    #
    # #continuer le train linear:
    # for i in range(3):
    #     linear_model_struct = LinearStructurer()
    #     linear_model_struct.nb_classes = 572
    #     linear_model_struct.input_shape = (images_size, images_size, 3)
    #     model = create_linear_model(linear_model_struct)
    #     description = getLinStructAsString(linear_model_struct)
    #
    #     model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, LINEAR_MODEL, images_size)
    #     model_name = "0_1_16_linear_model_1_10.h5"
    #     model = load_saved_model(model_path, model_name)
    #
    #     model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches,nb_test_batches, validation_split, epochs, batch_size=batch_size)
    #     test_model_batch(LINEAR_MODEL, model, model_name_root, my_batch, images_size, 1)
    #     epochs = epochs * 2
    #     clear_session()
    #
    #

    # images_size = 64
    # nb_train_batches = 2
    # nb_test_batches = 1
    # validation_split = 0.2
    # epochs = 1000
    #
    # for i in range(3):
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
    #     model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs, batch_size=batch_size)
    #     test_model_batch(LINEAR_MODEL, model, model_name_root, 0, images_size, 1)
    #     epochs = epochs * 2
    #     clear_session()
    #
    # images_size = 128
    # nb_train_batches = 5
    # nb_test_batches = 2
    # validation_split = 0.2
    # epochs = 1000
    # batch_size = 1024
    # for i in range(3):
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
    #     model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs, batch_size=batch_size)
    #     test_model_batch(LINEAR_MODEL, model, model_name_root, my_batch, images_size, nb_test_batches)
    #     epochs = epochs * 2
    #     clear_session()
    #
    # images_size = 256
    # nb_train_batches = 20
    # nb_test_batches = 5
    # validation_split = 0.2
    # epochs = 1000
    # batch_size = 512
    # for i in range(3):
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
    #     model_name_root = train_model_batch(LINEAR_MODEL, model, description, my_batch, images_size, nb_train_batches,
    #                                         nb_test_batches, validation_split, epochs, batch_size=batch_size)
    #     test_model_batch(LINEAR_MODEL, model, model_name_root, my_batch, images_size, nb_test_batches)
    #     epochs = epochs * 2
    #     clear_session()
#------------------------------------------------------------------------------------------------------------------------------------#
    # images_size = 64
    # nb_train_batches = 2
    # nb_test_batches = 1
    # epochs = 2000
    # batch_size = 2048
    # 
    # set_random_seed(1)
    # mlp_struct = MlpStructurer()
    # mlp_struct.nb_hidden_layers = 5
    # mlp_struct.nb_classes = 572
    # mlp_struct.input_shape = (images_size, images_size, 3)
    # mlp_struct.layers_size = [32, 32, 32, 32, 32]
    # model = create_custom_mlp(mlp_struct)
    # description = getMlpStructAsString(mlp_struct)
    #
    # model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, MLP, images_size)
    # model_name = "1_64_mlp_1_3.h5"
    # model = load_saved_model(model_path, model_name)
    #
    # model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                             nb_test_batches, validation_split, epochs, batch_size=batch_size)
    # test_model_batch(MLP, model, model_name_root, my_batch, images_size, nb_test_batches)
    # clear_session()
    # # #10/05/20 17h44-----------------------------------------------------------------------------------------------------------------
    # #
    # images_size = 64
    # nb_train_batches = 2
    # nb_test_batches = 1
    # epochs = 2000
    # batch_size = 2048
    # set_random_seed(1)
    # mlp_struct = MlpStructurer()
    # mlp_struct.nb_hidden_layers = 5
    # mlp_struct.nb_classes = 572
    # mlp_struct.input_shape = (images_size, images_size, 3)
    # mlp_struct.layers_size = [32, 32, 32, 32, 32]
    # model = create_custom_mlp(mlp_struct)
    # description = getMlpStructAsString(mlp_struct)
    #
    # model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, MLP, images_size)
    # model_name = "1_64_mlp_1_2.h5"
    # model = load_saved_model(model_path, model_name)
    #
    # model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                     nb_test_batches, validation_split, epochs, batch_size=batch_size)
    # test_model_batch(MLP, model, model_name_root, my_batch, images_size, nb_test_batches)
    # clear_session()
    # #----------------------------------------------------------------------------------------------------------------------------------
    # images_size = 64
    # nb_train_batches = 2
    # nb_test_batches = 1
    # epochs = 2000
    # batch_size = 2048
    # set_random_seed(1)
    # mlp_struct = MlpStructurer()
    # mlp_struct.nb_hidden_layers = 5
    # mlp_struct.nb_classes = 572
    # mlp_struct.input_shape = (images_size, images_size, 3)
    # mlp_struct.layers_size = [32, 32, 32, 32, 32]
    # model = create_custom_mlp(mlp_struct)
    # description = getMlpStructAsString(mlp_struct)
    #
    # model_path = '{}{}\\{}\\'.format(FIT_MODELS_DIR, MLP, images_size)
    # model_name = "1_64_mlp_1_1.h5"
    # model = load_saved_model(model_path, model_name)
    #
    # model_name_root = train_model_batch(MLP, model, description, my_batch, images_size, nb_train_batches,
    #                                     nb_test_batches, validation_split, epochs, batch_size=batch_size)
    # test_model_batch(MLP, model, model_name_root, my_batch, images_size, nb_test_batches)
    # clear_session()
    #----------------------------------------------------------------------------------------------------------------------------------
    # images_size = 256
    # nb_train_batches = 19
    # nb_test_batches = 4
    # epochs = 50
    # batch_size =32
    # #---------------CNN------
    # cnn_struct=CNNStructurer()
    # cnn_struct.nb_classes = 572
    # cnn_struct.nb_Conv2D_layers = 3  # nombre de couches cachées
    # cnn_struct.Conv2D_size_layers = [ (32, 3), (32, 3),(32, 3)]  # [input,filter_dimension] dans l'appel on utilisera un couple ( filter_dimension,filter_dimension)
    # cnn_struct.MaxPooling2D_use=True
    # cnn_struct.MaxPooling2D_Position = [2]  # Positionnement des couches Max2Pooling
    # cnn_struct.MaxPooling2D_values = 3 # valeur du filtre Max2Pooling
    # cnn_struct.use_l1l2_regularisation_Convolution_layers=True
    # cnn_struct.l1_value=0.15
    # cnn_struct.l2_value=0.5
    # cnn_struct.regul_kernel_indexes=[3]
    # #-----------------------------------------------
    # model = create_CNN_model(cnn_struct,images_size)
    # description = getcnnStructAsString(cnn_struct)
    # #------------------------------------------------
    # #epochs=300
    # model_name_root = train_model_batch(CNN, model, description, my_batch, images_size, nb_train_batches,
    #                                     nb_test_batches, validation_split, epochs, batch_size=batch_size)
    # test_model_batch(CNN, model, model_name_root, my_batch, images_size, nb_test_batches)
    # clear_session()
    #---------------------------------------------32X32-----------------------------------------------------------------

    # ---------------------------------------------64X64-----------------------------------------------------------------
    # images_size = 64
    # nb_train_batches = 2
    # nb_test_batches = 1
    # epochs = 100
    # batch_size = 512
    # # ---------------CNN------
    # cnn_struct = CNNStructurer()
    # cnn_struct.nb_classes = 572
    # cnn_struct.nb_Conv2D_layers = 5  # nombre de couches cachées
    # cnn_struct.Conv2D_size_layers = [(64, 3), (64, 3), (64, 3), (64, 3), (64, 3)]
    # cnn_struct.Conv2D_activation = 'selu'
    # cnn_struct.output_activation = 'softmax'
    # cnn_struct.MaxPooling2D_use = True
    # cnn_struct.MaxPooling2D_Position = [1, 3]  # Positionnement des couches Max2Pooling
    # cnn_struct.MaxPooling2D_values = 3  # valeur du filtre Max2Pooling
    # cnn_struct.use_l1l2_regularisation_Convolution_layers = True
    # cnn_struct.l1_value = 0.012
    # cnn_struct.l2_value = 0.030
    # cnn_struct.regul_kernel_indexes = [1, 4]
    # cnn_struct.loss = 'sparse_categorical_crossentropy'
    # cnn_struct.metrics = ['sparse_categorical_accuracy']
    # cnn_struct.use_dropout = True
    # cnn_struct.dropout_indexes = [0]
    # cnn_struct.dropout_value = 0.025
    # # -----------------------------------------------
    # model = create_CNN_model(cnn_struct, images_size)
    # description = getcnnStructAsString(cnn_struct)
    # # ------------------------------------------------
    # # epochs=300
    # model_name_root = train_model_batch(CNN, model, description, my_batch, images_size, nb_train_batches,
    #                                     nb_test_batches, validation_split, epochs, batch_size=batch_size)
    # test_model_batch(CNN, model, model_name_root, my_batch, images_size, nb_test_batches)
    # clear_session()
    #-------------------------------------------------------------------------------------------------------------------------
    images_size = 64
    nb_train_batches = 2
    nb_test_batches = 1
    epochs = 100
    batch_size = 64
    #----------------------------------------------------------------------------------------------------------------------------
    #name=32_resnet34_3_20.h5;validation_split=0.2;filters=32;nb_hidden=5;kernel=(3 3);batch_size=256;input_shape=32 32 3;
    # actication_layer=selu;out_activation=softmax;skip=True;nb_skip=2;Drop_out=True;index_dopout=[];drop_out_value=0.02;L1L2_layer=True;
    # L1L2_out=False;L1=0.012;L2=0.01;regul_inex=[1,4];use_maxpool=True;maxpoll_index=[1,3];loss=sparse_categorical_crossentropy;optim=Adam;
    # metrics=['sparse_categorical_accuracy'];padding=same;epochs=75
    #-----------------------------RESNET 64x64-------------------------------------------------------------------------
    # #---------------------------------------------128x128---------------------------------------------------------------

    # ---------------RSNT------
    rsnt=RsnetStructurer()
    rsnt.filters=32
    rsnt.nb_classes=572
    rsnt.nb_hidden_layers=5
    rsnt.kernel_size=(3,3)
    rsnt.input_shape=(64,64,3)
    rsnt.layers_activation='selu'
    rsnt.output_activation='softmax'
    rsnt.use_skip=True
    rsnt.nb_skip=2
    rsnt.use_dropout=True
    rsnt.dropout_value=0.03
    rsnt.dropout_indexes=[]
    rsnt.use_l1l2_regularisation_hidden_layers=True
    rsnt.use_l1l2_regularisation_output_layer=False
    rsnt.l1_value=0.015
    rsnt.l2_value=0.012
    rsnt.regulization_indexes=[1,4]
    rsnt.use_MaxPooling2D=True
    rsnt.MaxPooling2D_position=[1,3]
    rsnt.loss='sparse_categorical_crossentropy'
    rsnt.metrics=['sparse_categorical_accuracy']
    rsnt.padding='same'


    # -----------------------------------------------
    model = create_model_resenet34(rsnt)
    description = getResetStructAsString(rsnt)
    # ------------------------------------------------
    # epochs=300
    model_name_root = train_model_batch(RESNET34, model, description, my_batch, images_size, nb_train_batches,
                                        nb_test_batches, validation_split, epochs, batch_size=batch_size)
    test_model_batch(RESNET34, model, model_name_root, my_batch, images_size, nb_test_batches)
    clear_session()

    #-----------------------------32x32 7--------------------------------------
    images_size = 128
    nb_train_batches = 1
    nb_test_batches = 1
    epochs = 100
    batch_size = 256
    # ---------------CNN------
    cnn_struct = CNNStructurer()
    cnn_struct.nb_classes = 572
    cnn_struct.nb_Conv2D_layers = 4  # nombre de couches cachées
    cnn_struct.Conv2D_size_layers = [(128, 3), (128, 3), (128, 3), (128, 3)]
    cnn_struct.Conv2D_activation = 'softmax'
    cnn_struct.output_activation = 'selu'
    cnn_struct.MaxPooling2D_use = True
    cnn_struct.MaxPooling2D_Position = [1, 3]  # Positionnement des couches Max2Pooling
    cnn_struct.MaxPooling2D_values = 3  # valeur du filtre Max2Pooling
    cnn_struct.use_l1l2_regularisation_Convolution_layers = True
    cnn_struct.l1_value = 0.012
    cnn_struct.l2_value = 0.040
    cnn_struct.regul_kernel_indexes = [1]
    cnn_struct.loss = 'categorical_crossentropy'
    cnn_struct.metrics = ['categorical_accuracy']
    cnn_struct.use_dropout = True
    cnn_struct.dropout_indexes = [0]
    cnn_struct.dropout_value = 0.025
    # -----------------------------------------------
    model = create_CNN_model(cnn_struct, images_size)
    description = getcnnStructAsString(cnn_struct)
    # ------------------------------------------------
    # epochs=300
    model_name_root = train_model_batch(CNN, model, description, my_batch, images_size, nb_train_batches,
                                        nb_test_batches, validation_split, epochs, batch_size=batch_size)
    test_model_batch(CNN, model, model_name_root, my_batch, images_size, nb_test_batches)
    clear_session()


    # #---------------------------------------------128x128---------------------------------------------------------------
    # images_size = 128
    # nb_train_batches = 5
    # nb_test_batches = 2
    # epochs = 100
    # batch_size = 256
    # # ---------------CNN------
    # cnn_struct = CNNStructurer()
    # cnn_struct.nb_classes = 572
    # cnn_struct.nb_Conv2D_layers = 5  # nombre de couches cachées
    # cnn_struct.Conv2D_size_layers = [(32, 3), (64, 3), (128, 3), (64, 3), (32, 3)]
    # cnn_struct.Conv2D_activation = 'selu'
    # cnn_struct.output_activation = 'softmax'
    # cnn_struct.MaxPooling2D_use = True
    # cnn_struct.MaxPooling2D_Position = [1, 3]  # Positionnement des couches Max2Pooling
    # cnn_struct.MaxPooling2D_values = 3  # valeur du filtre Max2Pooling
    # cnn_struct.use_l1l2_regularisation_Convolution_layers = True
    # cnn_struct.l1_value = 0.012
    # cnn_struct.l2_value = 0.015
    # cnn_struct.regul_kernel_indexes = [1, 4]
    # cnn_struct.loss = 'sparse_categorical_crossentropy'
    # cnn_struct.metrics = ['sparse_categorical_accuracy']
    # cnn_struct.use_dropout = True
    # cnn_struct.dropout_indexes = [0]
    # cnn_struct.dropout_value = 0.02
    # # -----------------------------------------------
    # model = create_CNN_model(cnn_struct, images_size)
    # description = getcnnStructAsString(cnn_struct)
    # # ------------------------------------------------
    # # epochs=300
    # model_name_root = train_model_batch(CNN, model, description, my_batch, images_size, nb_train_batches,
    #                                     nb_test_batches, validation_split, epochs, batch_size=batch_size)
    # test_model_batch(CNN, model, model_name_root, my_batch, images_size, nb_test_batches)
    # clear_session()
    # # ---------------------------------------------256x256--------------------------------------------------------------
    # images_size = 256
    # nb_train_batches = 19
    # nb_test_batches = 4
    # epochs = 100
    # batch_size = 64
    # # ---------------CNN------
    # cnn_struct = CNNStructurer()
    # cnn_struct.nb_classes = 572
    # cnn_struct.nb_Conv2D_layers = 5  # nombre de couches cachées
    # cnn_struct.Conv2D_size_layers = [(32, 3), (64, 3), (128, 3), (64, 3), (32, 3)]
    # cnn_struct.Conv2D_activation = 'selu'
    # cnn_struct.output_activation = 'softmax'
    # cnn_struct.MaxPooling2D_use = True
    # cnn_struct.MaxPooling2D_Position = [1, 3]  # Positionnement des couches Max2Pooling
    # cnn_struct.MaxPooling2D_values = 3  # valeur du filtre Max2Pooling
    # cnn_struct.use_l1l2_regularisation_Convolution_layers = True
    # cnn_struct.l1_value = 0.012
    # cnn_struct.l2_value = 0.015
    # cnn_struct.regul_kernel_indexes = [1, 4]
    # cnn_struct.loss = 'sparse_categorical_crossentropy'
    # cnn_struct.metrics = ['sparse_categorical_accuracy']
    # cnn_struct.use_dropout = True
    # cnn_struct.dropout_indexes = [0]
    # cnn_struct.dropout_value = 0.025
    # # -----------------------------------------------
    # model = create_CNN_model(cnn_struct, images_size)
    # description = getcnnStructAsString(cnn_struct)
    # # ------------------------------------------------
    # model_name_root = train_model_batch(CNN, model, description, my_batch, images_size, nb_train_batches,
    #                                     nb_test_batches, validation_split, epochs, batch_size=batch_size)
    # test_model_batch(CNN, model, model_name_root, my_batch, images_size, nb_test_batches)
    # clear_session()

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
