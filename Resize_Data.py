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
from sklearn.model_selection import train_test_split, cross_val_score

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

'''for k in range(8, 12):
	plt.figure(k)
	for i in range(k*100, (k+1)*100):
		plt.subplot(10,10,i-k*100+1)
		plt.imshow(resized_data[i], cmap='Greys')
	plt.show()'''


resized_data = torch.unsqueeze(resized_data, dim=1)  # add one dimension--1
print(resized_data.shape)
labels = torch.Tensor(labels)

for i in range(len(labels)):
    labels[i] = labels[i] - 1

train_dataset = []
for i in range(len(resized_data)):
	train_dataset.append((resized_data[i],labels[i]))

print(type(labels))
print(len(labels))




train_loader, test_loader = train_test_split(train_dataset, test_size = .1)

train_loader = DataLoader(dataset=train_loader, batch_size=576, shuffle=True)
test_loader = DataLoader(dataset=test_loader, batch_size=640, shuffle=False)



for i, (images, labels) in enumerate(test_loader):
        if i == 1:
            break
        print(images.shape)
        print(labels.shape)




num_epochs = 15
num_classes = 8
learning_rate = 0.0005



# Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(12 * 12 * 128, 1000)
        self.fc2 = nn.Linear(1000, 8)

        '''for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')'''
# Forward

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

#Train the model
model = ConvNet()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Train loop
# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs): #epoch is 1
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels.long())
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted.long() == labels.long()).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))



# Test the model
model.eval() 
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.long() == labels.long()).sum().item()
    print("accuracy: "+ str(correct/total))



