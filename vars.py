DATASET_DIR = '..\\..\\dataset\\'
IMAGES_DIR = DATASET_DIR + 'train\\'
METADATA_FILE = DATASET_DIR + 'iwildcam2020_train_annotations.json'
IMAGES_CATEGORIES_PATH = DATASET_DIR + 'all_shuffled_images_with_category.csv'
IMAGE_CLASSES_REPRESENTATION = DATASET_DIR + 'all_shuffled_images_with_ytrain_representation.csv'
BATCHES_DIR = DATASET_DIR + 'batches\\'
BATCHES_FILES_ROOT = 'batch_'

TARGET_DIR = '..\\Target\\'
LOGS_DIR = TARGET_DIR + 'logs\\'
FIT_MODELS_DIR = TARGET_DIR + 'saved_models\\'
TRAIN_HYSTORY_DIR = TARGET_DIR + 'train_history\\'

NB_CLASSES = 572
NB_BATCHES = 200
IMAGE_HEIGHT = 108
IMAGE_LENGTH = 192