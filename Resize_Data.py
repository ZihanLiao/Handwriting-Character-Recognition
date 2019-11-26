import matplotlib.pyplot as plt
import pickle
import numpy as np
from torchvision.transforms import transforms
import torch

# Define loading function
def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


train_data = load_pkl('train_data.pkl')

labels = np.load('finalLabelsTrain.npy')


def resize_data_image(data):
    if len(data) != 48:  # # of row
        if len(data) < 48:
            if (48 - len(data)) % 2 != 0:
                data = np.pad(data, [(((48 - len(data)) // 2) + 1, (48 - len(data)) // 2), (0, 0)], mode='constant')
            else:
                data = np.pad(data, [((48 - len(data)) // 2, (48 - len(data)) // 2), (0, 0)], mode='constant')
        else:
            for i in range(len(data)):
                if i >= 48:
                    data = np.delete(data, 48, 0)
    if len(data[0]) != 48:  # # of column
        if len(data[0]) < 48:
            if (48 - len(data[0])) % 2 != 0:
                data = np.pad(data, [(0, 0), (((48 - len(data[0])) // 2) + 1, (48 - len(data[0])) // 2)],
                              mode='constant')
            else:
                data = np.pad(data, [(0, 0), (((48 - len(data[0])) // 2), (48 - len(data[0])) // 2)], mode='constant')
        else:
            for i in range(len(data[0])):
                if i >= 48:
                    data = np.delete(data, 48, 1)
    return data


resized_data = []

for i in range(len(train_data)):
    resized_data.append(resize_data_image(train_data[i]))
    if np.shape(resized_data[i]) != (48, 48):
        print("WRONG!")


# transform into array, then transform to tensor, get the train_data_raw
resized_data = np.array(resized_data).astype(int)
train_data_raw = torch.Tensor(resized_data)
train_data_raw = train_data_raw.unsqueeze(1)
print(train_data_raw[0].shape)





