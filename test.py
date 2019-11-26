import pickle
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms



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
    if len(data) != 50:  # # of row
        if len(data) < 50:
            if (50 - len(data)) % 2 != 0:
                data = np.pad(data, [(((50 - len(data)) // 2) + 1, (50 - len(data)) // 2), (0, 0)], mode='constant')
            else:
                data = np.pad(data, [((50 - len(data)) // 2, (50 - len(data)) // 2), (0, 0)], mode='constant')
        else:
            for i in range(len(data)):
                if i >= 50:
                    data = np.delete(data, 50, 0)
    if len(data[0]) != 50:  # # of column
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

# transform into array, then transform to tensor, get the train_data_raw
resized_data = np.array(resized_data).astype(int)
trans = transforms.ToTensor()
train_data_raw = trans(resized_data)


# test 1
num_epochs = 1
num_classes = 10
batch_size = 100
learning_rate = 0.001

# cross validation
train_loader = DataLoader(dataset=train_data_raw, batch_size=batch_size, shuffle=True)  # shuffle or not
test_loader = DataLoader(dataset=train_data_raw, batch_size=batch_size, shuffle=False)

# Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.init.xavier_uniform(self.layer1.weight),  # initialize Weights of first layer----xavier_uniform
            self.layer1.bias.data.fill_(0.01),  # initialize Bias
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.init.xavier_uniform(self.layer2.weight),  # initialize Weights of second layer
            self.layer2.bias.data.fill_(0.01),  # Bias(not sure)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)
# Forward

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# Train the model
model = ConvNet()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)