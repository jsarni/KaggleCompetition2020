DATASET_DIR = '..\\..\\dataset\\'
IMAGES_DIR = DATASET_DIR + 'train\\'
METADATA_FILE = DATASET_DIR + 'iwildcam2020_train_annotations.json'
IMAGES_CATEGORIES_PATH = '..\\dataset\\all_shuffled_images_with_category.csv'
IMAGE_CLASSES_REPRESENTATION = DATASET_DIR + 'all_shuffled_images_with_ytrain_representation.csv'
BATCHES_DIR = '..\\dataset\\'
BATCHES_FILES_ROOT = 'batch_'

PICKELED_IMAGES_DIR = '..\\..\\pickeled_dataset\\pickeled_images\\'
PICKELED_BACTHES_DIR = '..\\..\\pickeled_dataset\\pickeled_batches\\'

TEST_IMAGES_DIR = '..\\..\\pickeled_dataset\\test\\'
TEST_BACTH_FILE = BATCHES_DIR + 'batch_test'

TARGET_DIR = '..\\Target\\'
LOGS_DIR = TARGET_DIR + 'logs\\'
FIT_MODELS_DIR = TARGET_DIR + 'saved_models\\'
TRAIN_HYSTORY_DIR = TARGET_DIR + 'train_history\\'
PREDICTIONS_DIR = TARGET_DIR + 'predictions\\'

NB_CLASSES = 572
NB_BATCHES = 4
IMAGE_HEIGHT = 108
IMAGE_LENGTH = 192