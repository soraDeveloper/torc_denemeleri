import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
# from torch.optim import adam as optim

device = torch.device("cuda")

# parametreler

input_size = 30
hidden_size = 500
num_classes = 2
num_epoch = 250

learning_rate = 1e-8

girdi, cikti = load_breast_cancer(return_X_y=True)
"""
print(girdi)
print(girdi.shape)
print(cikti)
print(cikti.shape)
"""

# torch arrayşne çevirme

train_input = torch.from_numpy(girdi).float()
train_output = torch.from_numpy(cikti)

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.lrelu = nn.LeakyReLU(negative_slope=0.02)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forwardp(self, input):
        outfc1 = self.fc1(input)
        outfc1relu = self.lrelu(outfc1)
        out = self.fc2(outfc1relu)
        return out

model = NeuralNet(input_size,hidden_size,num_classes)

lossf = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    outputs = model(train_input)
    loss = lossf(outputs, train_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch: [{}/{}]. Loss: {: .4f}'.format(epoch+1, num_epoch, loss.item()))



