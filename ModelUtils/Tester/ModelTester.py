from tensorflow.keras.callbacks import TensorBoard
from ModelUtils.Models.Common import generateModelID
from datetime import date
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from DatasetReader.DatasetReader import *
import os


def train_model(model_type, model, model_description, batch_number, images_size, nb_train_batches, nb_test_batches, validation_split,  epochs, batch_size=4096, save_model=True, save_image=True):
    cur_date = date.today().strftime("%Y%m%d")
    logs_dir = '{}{}\\{}\\'.format(LOGS_DIR, model_type, images_size)
    fit_models_dir = '{}{}\\{}\\'.format(FIT_MODELS_DIR, model_type, images_size)
    images_dir = fit_models_dir

    model_id = generateModelID(model_type)
    model_name = '{}_{}_{}'.format(images_size, model_type, model_id)
    model_image = '{}{}.png'.format(images_dir, model_name)
    model_test_results_image = '{}_{}_test_results.png'.format(images_dir, model_name)
    log_name = '{}{}'.format(logs_dir, model_name)
    model_path_for_saving = '{}\\{}.h5'.format(fit_models_dir, model_name)

    if save_image:
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        plot_model(model, model_image)

    batch_images_ids_list, batchs_images_labels_list = read_batch_content(batch_number)
    train_images_ids_list, train_images_labels_list, test_images_ids_list, test_images_labels_list = split_images_to_train_and_test(batch_images_ids_list, batchs_images_labels_list)
    train_images_ids_batches, train_images_labels_batches = split_to_batches(train_images_ids_list, train_images_labels_list, nb_train_batches)
    test_images_ids_batches, test_images_labels_batches = split_to_batches(test_images_ids_list, test_images_labels_list, nb_test_batches)

    print("********************************************* Started Training model *********************************************")

    nb_finished_batches = 0
    for i in range(nb_train_batches):
        print("------------------------------------ Train Batch {} on {} ------------------------------------".format(i + 1, nb_train_batches))

        cur_batch_ids_list = train_images_ids_batches[i]
        cur_batch_labels_list = train_images_labels_batches[i]

        print("------------------- Started Loading Batch Data -------------------")
        xtrain, ytrain = load_batch(images_size, cur_batch_ids_list, cur_batch_labels_list)

        print("------------------- Finished Loading Data and starting Training -------------------")
        fit_model_on_batch(model, xtrain, ytrain, validation_split, epochs, batch_size, log_name)
        nb_finished_batches += 1

    print("********************************************* Finished Training model *********************************************")

    if save_model:
        if not os.path.exists(fit_models_dir):
            os.makedirs(fit_models_dir)
        model.save(model_path_for_saving)


    print("********************************************* Started Testing model *********************************************")
    nb_finished_batches = 0
    test_batches_results = []

    for i in range(nb_test_batches):
        print("------------------------------------ Test Batch {} on {} ------------------------------------".format(i + 1, nb_test_batches))

        cur_batch_ids_list = test_images_ids_batches[i]
        cur_batch_labels_list = test_images_labels_batches[i]

        print("------------------- Started Loading Train Batch Data -------------------")
        xtest, ytest = load_batch(images_size, cur_batch_ids_list, cur_batch_labels_list)

        print("------------------- Finished Loading Data and starting Model evaluation on test data -------------------")
        test_accuracy = model.evaluate(xtest, ytest)
        test_batches_results.append([nb_finished_batches, test_accuracy[-1] * 100])

        nb_finished_batches += 1
    print("********************************************* Finished Testing model *********************************************")


    batches = [batch_result[0] for batch_result in test_batches_results]
    accuracies = [batch_result[1] for batch_result in test_batches_results]
    plt.xlabel('batch')
    plt.ylabel('accuracy')
    plt.ylim(0, 100)
    plt.plot(batches, accuracies)
    plt.savefig(model_test_results_image)


    model_descr = "{};{};{};{};{}\n".format(images_size, model_description, str(epochs), model_id, cur_date)
    with open("{}{}_history.csv".format(TRAIN_HYSTORY_DIR, model_type), "a") as f:
        f.write(model_descr)



def fit_model_on_batch(model, batch_train_ds, batch_train_labels, validation_split, epochs_p, batch_size_p, log):
    tensorboard_callback = TensorBoard(log_dir=log)

    model.fit(batch_train_ds,
              batch_train_labels,
              validation_split=validation_split,
              epochs=epochs_p,
              batch_size=batch_size_p,
              callbacks=[tensorboard_callback]
              )

def load_model(model_dir, model_name):
    model_path = model_dir + model_name
    return load_model(model_path)