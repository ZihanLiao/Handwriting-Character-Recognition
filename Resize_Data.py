#! /bin/env python3
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

def load_pkl(fname):
	with open(fname,'rb') as f:
		return pickle.load(f)

def save_pkl(fname,obj):
	with open(fname,'wb') as f:
		pickle.dump(obj,f)

train_data = load_pkl('train_data.pkl')  

labels = np.load('finalLabelsTrain.npy')


def resize_data_image(data):    
	if(len(data) !=  48):
		if(len(data)<48):
			if((50-len(data))%2 != 0):
				data = np.pad(data, [(((48-len(data))//2)+1, (48-len(data))//2), (0, 0)], mode='constant')
			else:
				data = np.pad(data, [((48-len(data))//2, (48-len(data))//2), (0, 0)], mode='constant')
		else:
			for i in range(len(data)):
				if(i >= 48):
					data = np.delete(data, 48, 0)
	if(len(data[0])!=48):
		if(len(data[0])<48):
			if((48-len(data[0]))%2 != 0):
				data = np.pad(data, [(0, 0), (((48-len(data[0]))//2)+1, (48-len(data[0]))//2)], mode='constant')
			else:
				data = np.pad(data, [(0, 0), (((48-len(data[0]))//2), (48-len(data[0]))//2)], mode='constant')
		else:
			for i in range(len(data[0])):
				if(i >= 48):
					data = np.delete(data, 48, 1)
	return data

resized_data = []

for i in range(len(train_data)):
	resized_data.append(resize_data_image(train_data[i]))
	if (np.shape(resized_data[i]) != (48,48)):
		print("WRONG!")



resized_data = np.array(resized_data)
resized_data = torch.Tensor(resized_data)
print(type(resized_data))
print(resized_data.shape)

resized_data = torch.unsqueeze(resized_data, dim=1)
print(resized_data.shape)
labels = torch.Tensor(labels)

train_dataset = []
for i in range(len(resized_data)):
	train_dataset.append((resized_data[i],labels[i]))

print(type(labels))
print(len(labels))


train_loader = DataLoader(dataset=train_dataset, batch_size=400, shuffle=True)


for i, (images, labels) in enumerate(train_loader):
        if i == 1:
            break
        print(images.shape)
        print(labels.shape)
