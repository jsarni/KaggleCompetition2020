import pathlib

import os
import pandas as pd
import numpy as np
import random
import re
from PIL import Image
import glob
import pickle
from varsMatthieu import *


def split_dataset():
    print('> Started splitting dataset to train and test batches')
    train_batches_list = [i for i in range(NB_BATCHES)]
    random.shuffle(train_batches_list)
    nb_test_batches = int(0.2 * NB_BATCHES)
    test_batch_list = []
    for i in range(nb_test_batches):
        selected_batch = random.choice(train_batches_list)
        test_batch_list.append(selected_batch)
        train_batches_list.remove(selected_batch)

    return train_batches_list, test_batch_list


def read_image(image_id):
    image_path_root = '{}{}.*'.format(IMAGES_DIR, image_id)
    image_path = glob.glob(image_path_root)[0]

    image = Image.open(image_path)
    image = image.resize((IMAGE_HEIGHT, IMAGE_LENGTH))
    image_np = np.array(image)
    return image_np


def read_batch_images(batch_number):
    print('> Started loading batch images')
    batch_path = '{}{}{}'.format(BATCHES_DIR, BATCHES_FILES_ROOT, batch_number)
    batch = pd.read_csv(batch_path)

    images_ids = batch['image_id'].to_list()

    images_list = []

    for image in images_ids:
        images_list.append(read_image(image))

    return np.array(images_list)


def int_list_from_str(string):
    clean_string = re.sub(r'[\[\] ]', "", string)
    return [int(i) for i in clean_string.split(',')]


def read_batch_labels(batch_number):
    print('> Started loading batch labels')
    batch_path = '{}{}{}'.format(BATCHES_DIR, BATCHES_FILES_ROOT, batch_number)
    batch = pd.read_csv(batch_path)

    labels_str = batch['category_id'].to_list()

    labels = np.array([[int(label_str)] for label_str in labels_str])
    return labels


def load_unsplitted_batch(batch_number):
    batch_images = read_batch_images(batch_number)
    batch_labels = read_batch_labels(batch_number)

    return batch_images, batch_labels


def load_splitted_batch(batch_number):
    batch_images, batch_labels = load_unsplitted_batch(batch_number)
    return split_images_to_train_and_test(batch_images, batch_labels)

def read_image_to_pickle(image_id):
    image_path_root = '{}{}.*'.format(IMAGES_DIR, image_id)
    image_path = glob.glob(image_path_root)[0]

    pathlib.Path(PICKELED_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(PICKELED_IMAGES_DIR + '16\\').mkdir(parents=True, exist_ok=True)
    pathlib.Path(PICKELED_IMAGES_DIR + '32\\').mkdir(parents=True, exist_ok=True)
    pathlib.Path(PICKELED_IMAGES_DIR + '64\\').mkdir(parents=True, exist_ok=True)
    pathlib.Path(PICKELED_IMAGES_DIR + '128\\').mkdir(parents=True, exist_ok=True)
    pathlib.Path(PICKELED_IMAGES_DIR + '256\\').mkdir(parents=True, exist_ok=True)


    image = Image.open(image_path)

    for new_size in [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)]:
        current_size_dir = new_size[0]
        new_pickeled_image_path = '{}{}\\{}'.format(PICKELED_IMAGES_DIR, current_size_dir, image_id)
        r_image = image.resize(new_size)
        r_image_np = np.array(r_image)

        with open(new_pickeled_image_path, 'wb') as pickled_image:
            pickle.dump(r_image_np, pickled_image)

def read_batch_images_to_pickle(batch_number):
    batch_path = '{}{}{}'.format(BATCHES_DIR, BATCHES_FILES_ROOT, batch_number)
    batch = pd.read_csv(batch_path)

    images_ids = batch['image_id'].to_list()

    nb_images = 1
    total_images = len(images_ids)
    for image in images_ids:
        print("=====> image {} / {}".format(nb_images, total_images))
        read_image_to_pickle(image)
        nb_images += 1


def read_batch_content(batch_number):
    batch_path = '{}{}{}'.format(BATCHES_DIR, BATCHES_FILES_ROOT, batch_number)
    batch = pd.read_csv(batch_path)

    images_ids = batch['image_id'].to_list()
    images_labels = batch['category_id'].to_list()

    return images_ids, images_labels


def unpickle_image(image_size, image_id):
    image_path = '{}{}\\{}'.format(PICKELED_IMAGES_DIR, image_size, image_id)
    with open(image_path, 'rb') as f:
        image = pickle.load(f)
    return image

def split_images_to_train_and_test(images, labels):
    print('> Started batch to train and test data')
    nb_test_samples = int(0.2 * len(images)) + 1
    starting_pos = random.randint(0, len(images) - nb_test_samples)

    test_images = images[starting_pos:(starting_pos + nb_test_samples)]
    test_labels = labels[starting_pos:(starting_pos + nb_test_samples)]

    train_labels = np.concatenate((labels[:starting_pos], labels[(starting_pos + nb_test_samples):]))
    train_images = np.concatenate((images[:starting_pos], images[(starting_pos + nb_test_samples):]))

    return train_images, train_labels, test_images, test_labels



def split_to_batches(images_list, labels_list, nb_batches):
    batch_nb_elements = int(len(images_list) / nb_batches) + 1
    images_batches = []
    labels_batches = []
    for i in range(nb_batches):
        images_batches.append(images_list[(i*batch_nb_elements):((i+1)*batch_nb_elements)])
        labels_batches.append(labels_list[(i * batch_nb_elements):((i + 1) * batch_nb_elements)])
    return images_batches, labels_batches


def load_batch(image_size, batch_images_list, batch_labels_list):

    batch_images_loaded = []
    nb_loaded = 0
    for image in batch_images_list:
        nb_loaded += 1
        print("=====> Loading image :", nb_loaded)
        batch_images_loaded.append(unpickle_image(image_size, image))

    batch_labels = np.array([[int(label_str)] for label_str in batch_labels_list])
    batch_images = np.array(batch_images_loaded)

    return batch_images, batch_labels

def reshape_images(pickeled_images_dataset_path, batch_path):
    batch = pd.read_csv(batch_path)

    images_ids = batch['image_id'].to_list()

    nb = 1
    for id in images_ids:
        print(nb)
        if nb > 0:
            for s in [256, 128, 64, 32, 16]:
                with open('{}{}\\{}'.format(pickeled_images_dataset_path, s, id), 'rb') as o:
                    old = pickle.load(o)

                newim = old.reshape((s, s, 3))

                with open('{}{}\\{}'.format(pickeled_images_dataset_path, s, id), 'wb') as n:
                    pickle.dump(newim, n)
        nb += 1

def pickle_batch(batch_number, images_size, nb_train_batches, nb_test_batches):
    print(
        "********************************** Started Pickeling Batch {} - Image size is {} **********************************".format(batch_number, images_size))

    batch_images_ids_list, batchs_images_labels_list = read_batch_content(batch_number)

    train_images_ids_list, train_images_labels_list, test_images_ids_list, test_images_labels_list = split_images_to_train_and_test(batch_images_ids_list, batchs_images_labels_list)


    train_images_ids_batches, train_images_labels_batches = split_to_batches(train_images_ids_list, train_images_labels_list, nb_train_batches)
    test_images_ids_batches, test_images_labels_batches = split_to_batches(test_images_ids_list,
                                                                           test_images_labels_list, nb_test_batches)

    if not os.path.exists(PICKELED_BACTHES_DIR + str(images_size)):
        os.makedirs(PICKELED_BACTHES_DIR + str(images_size))


    for i in range(nb_train_batches):
        print("------------------------------------ Pickeling Train Batch {} on {} ------------------------------------".format(
            i + 1, nb_train_batches))

        cur_batch_ids_list = train_images_ids_batches[i]
        cur_batch_labels_list = train_images_labels_batches[i]

        xtrain, ytrain = load_batch(images_size, cur_batch_ids_list, cur_batch_labels_list)

        cur_batch_images_path = "{}{}\\batch_{}_{}_train_images_{}".format(PICKELED_BACTHES_DIR, images_size, batch_number, images_size, i)
        cur_batch_labels_path = "{}{}\\batch_{}_{}_train_labels_{}".format(PICKELED_BACTHES_DIR, images_size, batch_number, images_size, i)
        with open(cur_batch_images_path, 'wb') as pickeled_batch_images:
            pickle.dump(xtrain, pickeled_batch_images)

        with open(cur_batch_labels_path, 'wb') as pickeled_batch_labels:
            pickle.dump(ytrain, pickeled_batch_labels)


    for i in range(nb_test_batches):
        print("------------------------------------ Pickeling Test Batch {} on {} ------------------------------------".format(
            i + 1, nb_test_batches))

        cur_batch_ids_list = test_images_ids_batches[i]
        cur_batch_labels_list = test_images_labels_batches[i]

        xtest, ytest = load_batch(images_size, cur_batch_ids_list, cur_batch_labels_list)

        cur_batch_images_path = "{}{}\\batch_{}_{}_test_images_{}".format(PICKELED_BACTHES_DIR, images_size, batch_number, images_size, i)
        cur_batch_labels_path = "{}{}\\batch_{}_{}_test_labels_{}".format(PICKELED_BACTHES_DIR, images_size, batch_number, images_size, i)

        with open(cur_batch_images_path, 'wb') as pickeled_batch_images:
            pickle.dump(xtest, pickeled_batch_images)

        with open(cur_batch_labels_path, 'wb') as pickeled_batch_labels:
            pickle.dump(ytest, pickeled_batch_labels)

def load_pickeled_batch(batch_type, dataset_part, batch_number, images_size):
    batch_images_path = '{}{}\\batch_{}_{}_{}_images_{}'.format(PICKELED_BACTHES_DIR, images_size, dataset_part, images_size, batch_type, batch_number)
    batch_labels_path = '{}{}\\batch_{}_{}_{}_labels_{}'.format(PICKELED_BACTHES_DIR, images_size, dataset_part, images_size, batch_type, batch_number)

    with open(batch_images_path, 'rb') as i:
        batch_images = pickle.load(i)

    with open(batch_labels_path, 'rb') as l:
        batch_labels = pickle.load(l)

    return batch_images, batch_labels