import json
import pandas as pd
import random
import pathlib
from vars import *

def prepare_train_labels_for_images_classes():
    path = METADATA_FILE
    with open(path, 'r') as f:
        file = json.load(f)

    annotations_json =json.dumps(file['annotations'])
    images_df = pd.read_json(annotations_json)[['image_id', 'category_id']]

    images_df = images_df.sample(frac=1).reset_index(drop=True)

    images_with_categories = images_df[['image_id', 'category_id']]
    images_with_categories.to_csv(IMAGES_CATEGORIES_PATH, index=False)


def split_dataset_to_batches():
    complete_dataset = pd.read_csv(IMAGES_CATEGORIES_PATH)
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


