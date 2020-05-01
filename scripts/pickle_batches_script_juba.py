from DatasetReader.DatasetReader_Juba import pickle_batch

if __name__ == '__main__':
    pickle_batch(0, 16, 1, 1)
    pickle_batch(0, 32, 1, 1)
    pickle_batch(0, 64, 2, 1)
    pickle_batch(0, 128, 5, 2)
    pickle_batch(0, 256, 20, 5)