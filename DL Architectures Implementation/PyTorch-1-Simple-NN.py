#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:06:53 2020

@author: rushirajsinhparmar
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# CREATE FULLY CONNECTED NETWORK 

#NOTE: The arguments to __init__() are meant to be added as you gradually procees
class NN(nn.Module):
    def __init__(self, input_size, num_classes): #(28,28)
        super(NN,self).__init__()   #Super calls the initialisation fn of Parent Class
        
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        return x
 
# SIMPLE TESTING
    
#model = NN(784,10) 
#x = torch.randn(64,784)   # TEST data: 64 = mini-batch size
#print(model(x).shape)
        
    
# SET DEVICE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Runnning on: ", device)

# HYPERPARAMETERS:

input_size = 784
num_classes = 10
learning_rate = 0.01
batch_size = 64
num_epochs = 2

# LOAD DATA:

train_dataset = datasets.FashionMNIST(
    root = '/Users/rushirajsinhparmar/Downloads/PyTorch/', train = True, transform = transforms.ToTensor(),
    download = False) # Keep download=True if you dont have the dataset

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = datasets.FashionMNIST(
    root = '/Users/rushirajsinhparmar/Downloads/PyTorch/', train = False, transform = transforms.ToTensor(),
    download = False) # Keep download=True if you dont have the dataset

test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# INITIALISE THE NETWORK:

model = NN(input_size = input_size, num_classes = num_classes).to(device)

# LOSS and OPTIMIZER

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# TRAIN the NETWORK

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        
        # Get data to cuda if possible
        data = data.to(device = device)
        targets = targets.to(device = device)
        
        #Get to correct shape
        data = data.reshape(data.shape[0], -1)  # Keep the 1st dimension (64) intact
        
        #forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        #backward:
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent or Adam step
        optimizer.step()
        
# Check accuracy on training and testing set
        
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")
    
    num_correct = 0
    num_samples = 0
    
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/(num_samples)*100:.2f}')
        
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)