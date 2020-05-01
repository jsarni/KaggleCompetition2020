from DatasetReader.DatasetReader_aghylas import pickle_batch

if __name__ == '__main__':
    pickle_batch(2, 16, 1, 1)
    pickle_batch(2, 32, 1, 1)
    pickle_batch(2, 64, 2, 1)
    pickle_batch(2, 128,5, 2)
    pickle_batch(2, 256,20, 5)