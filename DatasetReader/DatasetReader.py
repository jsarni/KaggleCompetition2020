import pathlib

import pandas as pd
import numpy as np
import random
import re
from PIL import Image,ImageFile
import glob
import pickle



from vars import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def split_batch_to_train_and_val(images, labels):
    print('> Started batch to train and validation data')
    nb_val_samples = int(0.2 * len(images)) + 1
    starting_pos = random.randint(0, len(images) - nb_val_samples)

    val_images = images[starting_pos:(starting_pos + nb_val_samples)]
    val_labels = labels[starting_pos:(starting_pos + nb_val_samples)]

    train_labels = np.concatenate((labels[:starting_pos, :], labels[(starting_pos + nb_val_samples):, :]))
    train_images = np.concatenate((images[:starting_pos, :], images[(starting_pos + nb_val_samples):, :]))

    return train_images, train_labels, val_images, val_labels


def load_unsplitted_batch(batch_number):
    batch_images = read_batch_images(batch_number)
    batch_labels = read_batch_labels(batch_number)

    return batch_images, batch_labels


def load_splitted_batch(batch_number):
    batch_images, batch_labels = load_unsplitted_batch(batch_number)
    return split_batch_to_train_and_val(batch_images, batch_labels)

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
        flattened_image = r_image_np.flatten()

        with open(new_pickeled_image_path, 'wb') as pickled_image:
            pickle.dump(flattened_image, pickled_image)

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
