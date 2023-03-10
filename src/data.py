import pickle
import numpy as np


def pickle_load(file_name):
    with open(file_name, 'rb') as f:
        file_name = f"./data/pickle{file_name}.pkl"
        return pickle.load(f)

def pickle_dump(file_name, data):
    file_name = f"./data/pickle{file_name}.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

