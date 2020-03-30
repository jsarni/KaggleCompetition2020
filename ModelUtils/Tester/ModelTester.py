from tensorflow.keras.callbacks import TensorBoard
from ModelUtils.Models.Common import generateModelID
from datetime import date
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from DatasetReader.DatasetReader import *
import os


def train_model(model_type, model, model_description, epochs, batch_size=4096, save_model=True, save_image=True):
    cur_date = date.today().strftime("%Y%m%d")
    logs_dir = '{}{}\\'.format(LOGS_DIR, model_type)
    fit_models_dir = '{}{}\\'.format(FIT_MODELS_DIR, model_type)
    images_dir = fit_models_dir

    model_id = generateModelID(model_type)
    model_name = '{}_{}'.format(model_type, model_id)
    model_image = '{}{}.png'.format(images_dir, model_name)
    model_test_results_image = '{}_{}_test_results.png'.format(images_dir, model_name)
    log_name = '{}{}'.format(logs_dir, model_name)
    model_path_for_saving = '{}{}\\{}.h5'.format(fit_models_dir, model_type, model_name)

    if save_image:
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        plot_model(model, model_image)

    train_batches_ids, test_batches_ids = split_dataset()

    print("********************************************* Started Training model *********************************************")

    nb_finished_batches = 0
    for train_batch in train_batches_ids:
        print("------------------------------------ Train Batch {} on {} ------------------------------------".format(nb_finished_batches, len(train_batches_ids)))

        batch_train_images, batch_train_labels, batch_val_images, batch_val_labels = load_splitted_batch(train_batch)

        fit_model_on_batch(model, batch_train_images, batch_train_labels, batch_val_images, batch_val_labels, epochs, batch_size, log_name)
        nb_finished_batches += 1

    print("********************************************* Finished Training model *********************************************")

    if save_model:
        if not os.path.exists(fit_models_dir):
            os.makedirs(fit_models_dir)
        model.save(model_path_for_saving)


    print("********************************************* Started Testing model *********************************************")
    nb_finished_batches = 0
    test_batches_results = []
    for test_batch in test_batches_ids:
        print("------------------------------------ Test Batch {} on {} ------------------------------------".format(nb_finished_batches, len(test_batches_ids)))

        batch_images, batch_labels = load_unsplitted_batch(test_batch)

        test_accuracy = model.evaluate(batch_images, batch_labels)
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


    model_descr = "{};{};{};{}\n".format(model_description, str(epochs), model_id, cur_date)
    with open("{}{}_history.csv".format(TRAIN_HYSTORY_DIR, model_type), "a") as f:
        f.write(model_descr)



def fit_model_on_batch(model, batch_train_ds, batch_train_labels, batch_val_ds, batch_val_labels, epochs_p, batch_size_p, log):
    tensorboard_callback = TensorBoard(log_dir=log)

    model.fit(batch_train_ds,
              batch_train_labels,
              validation_data=(batch_val_ds, batch_val_labels),
              epochs=epochs_p,
              batch_size=batch_size_p,
              callbacks=[tensorboard_callback]
              )
