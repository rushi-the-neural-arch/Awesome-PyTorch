from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from datetime import datetime

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import itertools
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

start = datetime.now()


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)

    def forward(self,t):

        # (1) Input layer
        t = t

        # (2) First Hidden layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        # (3) Second Hidden layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        # (4) 1st Linear Layer
        t = t.reshape(-1, 12*4*4)  # OR t = t.flatten(start_dim = 1)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) Second Linear Layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) Third Linear Layer
        t = self.out(t)

        return t



def get_num_correct(preds,labels):
    return preds.argmax(dim = 1).eq(labels).sum().item()



train_set = torchvision.datasets.FashionMNIST(
    root = '/Users/rushirajsinhparmar/Downloads/PyTorch/',
    train = True,
    download = False,
    transform = transforms.Compose([transforms.ToTensor()])
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = True)

########## MAIN TENSORBOARD TASK STARTS NOW

tb = SummaryWriter()       

network = Network()

images,labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)


tb.add_image('images', grid)
tb.add_graph(network, images)

####### RUN :    tensorboard --logdir=runs     

########################################### TRAINING LOOP #############################################

optimizer = optim.Adam(network.parameters(), lr = 0.01)

for epoch in range(3):

    total_loss = 0
    total_correct = 0

    for batch in train_loader: 

        images, labels = batch

        preds = network(images)
        loss = F.cross_entropy(preds,labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)

    tb.add_scalar("Loss", total_loss, epoch)
    tb.add_scalar("Number Correct", total_correct, epoch)
    tb.add_scalar("Accuracy", total_correct /len(train_set), epoch)

    # tb.add_histogram("Conv1.bias", network.conv1.bias, epoch)
    # tb.add_histogram("Conv1.weight", network.conv1.weight, epoch)
    # tb.add_histogram("Conv1.weight.grad", network.conv1.weight.grad, epoch)

    # BUT, the above histogram addition is hard coded. We can automate it by including all network parameters

    for name, weight in network.named_parameters():
        tb.add_histogram(name,weight, epoch)
        tb.add_histogram(f'{name}.grad', weight.grad, epoch)
    
    tb.add_histogram("Conv2.weight", network.conv2.weight, epoch)


    print("Epoch: ", epoch, "Loss: ", total_loss, "total_corect: ", total_correct)




tb.close()

print("Time Taken: ", datetime.now() - start)


# IN SHORT : Code Snippet for TENSORBOARD --------------------------------

'''

optimizer = optim.Adam(network.parameters(), lr = 0.01)

tb = SummaryWriter()       

network = Network()

images,labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)

tb.add_image('images', grid)
tb.add_graph(network, images)

for epoch in range(1):

    total_loss = 0
    total_correct = 0

    for batch in train_loader: 

        images, labels = batch

        preds = network(images)
        loss = F.cross_entropy(preds,labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)

    tb.add_scalar("Loss", total_loss, epoch)
    tb.add_scalar("Number Correct", total_correct, epoch)
    tb.add_scalar("Accuracy", total_correct /len(train_set), epoch)

    tb.add_histogram("Conv1.bias", network.conv1.bias, epoch)
    tb.add_histogram("Conv1.weight", network.conv1.weight, epoch)
    tb.add_histogram("Conv1.weight.grad", network.conv1.weight.grad, epoch)
    
    tb.add_histogram("Conv2.weight", network.conv2.weight, epoch)


    print("Epoch: ", epoch, "Loss: ", total_loss, "total_corect: ", total_correct)


tb.close()

'''