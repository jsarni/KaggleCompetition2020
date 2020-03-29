import json
import pandas as pd
import numpy as np
import random
import pathlib
import re
from PIL import Image
import glob
from DatasetReader.vars import *

def prepare_train_labels_for_images_classes():
    path = DATASET_DIR + METADATA_FILE
    with open(path, 'r') as f:
        file = json.load(f)

    annotations_json =json.dumps(file['annotations'])
    images_df = pd.read_json(annotations_json)[['image_id', 'category_id']]

    images_df['ytrain_row'] = images_df.apply(lambda row: [-1 for i in range(NB_CLASSES)], axis=1)

    for index, row in images_df.iterrows():
        row['ytrain_row'][row['category_id']] = 1

    images_df = images_df.sample(frac=1).reset_index(drop=True)

    images_with_categories = images_df[['image_id', 'category_id']]
    images_with_ytrain_rep = images_df[['image_id', 'ytrain_row']]

    images_with_ytrain_rep.to_csv(IMAGES_CATEGORIES_PATH, index=False)
    images_with_categories.to_csv(IMAGE_CLASSES_REPRESENTATION, index=False)


def split_dataset_to_batches():
    complete_dataset = pd.read_csv(IMAGE_CLASSES_REPRESENTATION)
    pathlib.Path(BATCHES_DIR).mkdir(parents=True, exist_ok=True)

    nb_samples = complete_dataset.shape[0]
    nb_samples_per_batch = int(nb_samples / NB_BATCHES) + 1
    remaining_batch_numbers = [x for x in range(NB_BATCHES)]
    for i in range(NB_BATCHES):
        random_batch_number = random.choice(remaining_batch_numbers)
        remaining_batch_numbers.remove(random_batch_number)

        current_batch_path = '{}{}{}'.format(BATCHES_DIR, BATCHES_FILES_ROOT, str(random_batch_number))
        current_batch = complete_dataset[(i * nb_samples_per_batch): ((i + 1) * nb_samples_per_batch)]
        current_batch.to_csv(current_batch_path, index=False)


def split_dataset():
    batches_list = [i for i in range(NB_BATCHES)]
    random.shuffle(batches_list)
    print(type(batches_list))
    nb_test_batches = int(0.2 * NB_BATCHES)
    test_batch_list = []
    for i in range(nb_test_batches):
        selected_batch = random.choice(batches_list)
        test_batch_list.append(selected_batch)
        batches_list.remove(selected_batch)

    return batches_list, test_batch_list


def int_list_from_str(string):
    clean_string = re.sub(r'[\[\] ]', "", string)
    return [int(i) for i in clean_string.split(',')]


def read_batch_ytrain(batch_number):
    batch_path = '{}{}{}'.format(BATCHES_DIR, BATCHES_FILES_ROOT, batch_number)
    batch = pd.read_csv(batch_path)

    Y_str = batch['ytrain_row'].to_list()

    nb_val_samples = int(0.2 * len(Y_str))
    starting_pos = random.randint(0, len(Y_str) - nb_val_samples)

    YVal_str = Y_str[starting_pos:(starting_pos + nb_val_samples)]
    YTrain_str = Y_str[0:starting_pos] + Y_str[(starting_pos + nb_val_samples):]

    YTrain = np.array(list(map(int_list_from_str, YTrain_str)))
    YVal = np.array(list(map(int_list_from_str, YVal_str)))

    return YTrain, YVal


def read_batch_ytest(batch_number):
    batch_path = '{}{}{}'.format(BATCHES_DIR, BATCHES_FILES_ROOT, batch_number)
    batch = pd.read_csv(batch_path)

    Y_str = batch['ytrain_row'].to_list()

    YTest = np.array(list(map(int_list_from_str, Y_str)))

    return YTest


def read_image(image_id):
    image_path_root = '{}{}.*'.format(IMAGES_DIR, image_id)
    image_path = glob.glob(image_path_root)[0]

    image = Image.open(image_path)
    image = image.resize(IMAGES_FIXED_SIZE)
    image_np = np.array(image)
    flattened_image = image_np.flatten()
    return flattened_image


def read_batch_images(batch_number):
    batch_path = '{}{}{}'.format(BATCHES_DIR, BATCHES_FILES_ROOT, batch_number)
    batch = pd.read_csv(batch_path)

    images_ids = batch['image_id'].to_list()

    images_list = []

    for image in images_ids:
        images_list.append(read_image(image))

    return np.array(images_list)