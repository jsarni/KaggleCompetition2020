from DatasetReader.DatasetPreparation import *

if __name__ == '__main__':
    print("------------------------------ Preparing Dataset ------------------------------")
    prepare_train_labels_for_images_classes()
    print("------------------------------ Splitting Dataset into Batches ------------------------------")
    split_dataset_to_batches()