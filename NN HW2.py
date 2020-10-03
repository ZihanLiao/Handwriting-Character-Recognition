import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, cross_val_score
import re
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

class arguments():
    def __init__(self):
        self.n_input = 13
        self.hidden = 50
        self.n_epoch = 50
        self.lr = 1e-3
        self.batch_size = 1
        self.n_class = 3

# Model
class S_NN(nn.Module):
    def __init__(self):
        super(S_NN, self).__init__()
        self.hidden = nn.Linear(args.n_input, args.hidden)
        self.output = nn.Linear(args.hidden, args.n_class)

        self.LReLU = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


    def _forward(self, data):
        x = self.hidden(data)
        x = self.LReLU(x)
        # x = self.tanh(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


np.random.seed(5)

args = arguments()
model = S_NN()

array = np.zeros((14, 178))
f = open('winedata.txt', 'r')
lines = f.readlines()
for line in range(len(lines)):
    array[:, line] = [float(x) for x in re.findall(r"\d+\.?\d*", lines[line])]


d = array[1:array.shape[0]+1, :].T
d_new = stats.zscore(d, axis=0)

label_ = torch.from_numpy(array[0, :])-1
data_ = torch.from_numpy(d_new)
dataset = TensorDataset(data_, label_)

train_, test_ = train_test_split(dataset, test_size = .2)
train_data = DataLoader(dataset=train_, batch_size=args.batch_size, shuffle=True)
test_data = DataLoader(dataset=test_, batch_size=args.batch_size, shuffle=False)


'''Loss and optimizer'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


'''Training'''
total_step = len(train_data)
loss_list = []
acc_list = np.zeros((args.n_epoch, len(train_data)))
acc_last = np.zeros((1, args.n_epoch))

for epoch in range(args.n_epoch):
    for i, (data, label) in enumerate(train_data):
        '''Forward'''
        output = model._forward(data.float())
        loss = criterion(output, label.long())
        loss_list.append(loss.item())

        '''Backward'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''Tracking'''
        total = len(train_data)
        _, pre = torch.max(output, 1)
        correct = (pre.long() == label.long()).sum().item()
        acc_list[epoch, i] = correct
        # print(acc_list)

        '''Monitoring'''
        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, args.n_epoch, i + 1, total_step, loss.item(), (acc_list[epoch, :].sum() / total) * 100))

    acc_last[0, epoch] = acc_list[epoch, :].sum() / len(train_data)


plt.figure(1)
plt.plot(acc_last[0, :])
plt.title('Training Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')


'''test'''
model.eval()
with torch.no_grad():
    correct_test = 0
    total_test = 0
    for data_test, label_test in test_data:
        output_test = model._forward(data_test.float())
        _, pre_test = torch.max(output_test, 1)
        total_test += label_test.size(0)
        correct_test += (pre_test.long() == label_test.long()).sum().item()
    print('Accuracy on test data: ' + str(correct_test / total_test))