import matplotlib.pyplot as plt
import pickle
import numpy as np


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


train_data = load_pkl('train_data.pkl')

labels = np.load('finalLabelsTrain.npy')


def resize_data_image(data):
    if len(data) != 50:
        if len(data) < 50:
            if (50 - len(data)) % 2 != 0:
                data = np.pad(data, [(((50 - len(data)) // 2) + 1, (50 - len(data)) // 2), (0, 0)], mode='constant')
            else:
                data = np.pad(data, [((50 - len(data)) // 2, (50 - len(data)) // 2), (0, 0)], mode='constant')
        else:
            for i in range(len(data)):
                if i >= 50:
                    data = np.delete(data, 50, 0)
    if len(data[0]) != 50:
        if len(data[0]) < 50:
            if (50 - len(data[0])) % 2 != 0:
                data = np.pad(data, [(0, 0), (((50 - len(data[0])) // 2) + 1, (50 - len(data[0])) // 2)],
                              mode='constant')
            else:
                data = np.pad(data, [(0, 0), (((50 - len(data[0])) // 2), (50 - len(data[0])) // 2)], mode='constant')
        else:
            for i in range(len(data[0])):
                if i >= 50:
                    data = np.delete(data, 50, 1)
    return data


resized_data = []

for i in range(len(train_data)):
    resized_data.append(resize_data_image(train_data[i]))
    if np.shape(resized_data[i]) != (50, 50):
        print("WRONG!")
