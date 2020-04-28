import json
from pandas.io.json import json_normalize
import pandas as pd
import numpy as np
import random
import pathlib
import re
from PIL import Image
import glob
import matplotlib.pyplot as plt
import pickle


def prepare_test_images_ids(path, dest):
    with open(path, 'r') as f:
        file = json.load(f)

    annotations_json =json.dumps(file['images'])
    test_images = pd.read_json(annotations_json)[['id']]
    test_images.to_csv(dest, index=False)
    

prepare_test_images_ids('../dataset/iwildcam2020_test_information.json', '../pickeled_dataset/test_images_ids.csv')


def read_test_image_to_pickle(image_id):
    image_path_root = '{}{}.*'.format('../dataset/test/', image_id)
    image_path = glob.glob(image_path_root)[0]
    pickeled_test_images_path = '../pickeled_dataset/test/'
    pathlib.Path().mkdir(parents=True, exist_ok=True)
    pathlib.Path(pickeled_test_images_path + '16/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(pickeled_test_images_path + '32/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(pickeled_test_images_path + '64/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(pickeled_test_images_path + '128/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(pickeled_test_images_path + '256/').mkdir(parents=True, exist_ok=True)


    image = Image.open(image_path)

    for new_size in [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)]:
        current_size_dir = new_size[0]
        new_pickeled_image_path = '{}{}\\{}'.format(pickeled_test_images_path, current_size_dir, image_id)
        r_image = image.resize(new_size)
        r_image_np = np.array(r_image)
        with open(new_pickeled_image_path, 'wb') as pickled_image:
            pickle.dump(r_image_np, pickled_image)

images_ids = pd.read_csv('../pickeled_dataset/test_images_ids.csv')['id'].to_list()

nb_images = 0
total = len(images_ids)
for image in images_ids:
    nb_images += 1
    print("image {}/{} : {}".format(nb_images, total, image))
    read_test_image_to_pickle(image)