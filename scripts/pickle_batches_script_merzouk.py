from DatasetReader.DatasetReader_merzouk import pickle_batch

if __name__ == '__main__':
    pickle_batch(3, 16, 1, 1)
    pickle_batch(3, 32, 1, 1)
    pickle_batch(3, 64, 2, 1)
    pickle_batch(3, 128, 5, 2)
    pickle_batch(3, 256, 20, 5)