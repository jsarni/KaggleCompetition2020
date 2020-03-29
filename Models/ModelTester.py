from tensorflow.keras.callbacks import TensorBoard
from cifar10.models.Common import getRandomModelID
from datetime import date
from tensorflow.keras.utils import plot_model
import os


def test_models(model_type, models, models_descriptions, train_ds, train_labels, test_ds, test_labels, epochs_p, batch_size_p=4096, save_model=False, save_image=False):
    cur_date = date.today().strftime("%Y%m%d")
    logs_dir = '..\\..\\..\\logs\\{}\\{}\\'.format(model_type, cur_date)
    fit_models_dir = '..\\saved_models\\'.format(model_type.split("_")[0])
    images_dir = '..\\images\\'
    trained_models = 0

    for e in epochs_p:
        for i in range(len(models)):
            print("********************************************* Training model number {} *********************************************".format(e*i + i + 1))
            model_id = getRandomModelID()

            if save_image:
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                plot_model(models[i], '{}{}_{}.png'.format(images_dir, model_type, model_id))
            log_name = logs_dir + model_type + '_' + model_id + '.log'
            model_name = fit_models_dir + model_type + '_' + model_id + '.h5'
            tensorboard_callback = TensorBoard(log_dir=log_name)

            models[i].fit(train_ds,
                          train_labels,
                          validation_data=(test_ds, test_labels),
                          epochs=e,
                          batch_size=batch_size_p,
                          callbacks=[tensorboard_callback]

                          )

            if save_model:
                if not os.path.exists(fit_models_dir):
                    os.makedirs(fit_models_dir)
                models[i].save(model_name)

            train_accuracy = models[i].evaluate(train_ds, train_labels)
            val_accuracy = models[i].evaluate(test_ds, test_labels)
            print(val_accuracy[-1],train_accuracy[-1])
            model_descr = "{};{};{};{};{};{}\n".format(models_descriptions[i], str(e), model_id, str(train_accuracy[-1]), str(val_accuracy[-1]), cur_date)
            with open("..\\historique_tests\\tested_{}_history.csv".format(model_type), "a") as f:
                f.write(model_descr)

            trained_models += 1

    return trained_models


