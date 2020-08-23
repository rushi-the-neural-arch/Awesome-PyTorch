import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from datetime import datetime

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


train_set = torchvision.datasets.FashionMNIST(
    root = '/Users/rushirajsinhparmar/Downloads/PyTorch/',
    train = True,
    download = False,
    transform = transforms.Compose([transforms.ToTensor()])
)

network = Network()

def get_all_preds(model, loader):

    all_preds = torch.tensor([])

    for batch in loader:
        images, labels = batch
        preds = model(images)

        all_preds = torch.cat((all_preds,preds), dim = 0)

    return all_preds

start = datetime.now()

prediction_loader = torch.utils.data.DataLoader(train_set, batch_size = 32)

print("Making prediction with Gradient Tracking ON (by default)")

train_preds = get_all_preds(network, prediction_loader)

print("Time taken to predict with Gradient Tracking ON : {}".format(datetime.now() - start))

print("Shape of train_preds {}".format(train_preds.shape))


'''
 Currently PyTorch is keeping a track of Gradients ( Even though we are not training the model as of now)

This results into more time for predictimg 60K images and the memory consumption as well 

 You can turn this feature off gloably but there is also a local way to do that see below

'''

print(train_preds.requires_grad)   # True, as of now

train_preds.grad # Won't return anything! Check in Jupyter Notebook

train_preds.grad_fn 

#### NOW, we will turn off gradient tracking locally 

print("Making predictions by turning OFF Gradient Tracking")

print("This will take less time and consume less memory due to no accumulation of gradients")

start2 = datetime.now()

with torch.no_grad():
    
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size = 32)
    train_preds = get_all_preds(network, prediction_loader)

print("Time taken to predict with Gradient Tracking OFF : {}".format(datetime.now() - start2))

print("Notice the time difference") # 8 seconds less


# Difference between torch.cat and torch.stack - V.V.IMP !!
# https://stackoverflow.com/questions/54307225/whats-the-difference-between-torch-stack-and-torch-cat-functions 


                        ############## PLOTTING CONFUSION MATRIX ###################

print(" ******************         ##############          ****************\n\n\n")

print("Plotting Confusion Matrix\n\n")


torch.set_printoptions(linewidth = 120)

print(train_set.targets)

print(train_preds.argmax(dim = 1))

stacked = torch.stack((train_set.targets, train_preds.argmax(dim = 1)), dim = 1)

print("STACKED TENSOR: ", stacked)

print(stacked[0].tolist())

cmt = torch.zeros((10,10), dtype = torch.int32)    # (10,10) because we have 10 classes

for p in stacked:

    tl, pl = p.tolist()
    cmt[tl,pl] = cmt[tl, pl] + 1

print("Confusion Matrix:", cmt)


# NOTE: V.IMP - Refer Juputer Notebook for better understanding:https://colab.research.google.com/drive/1Ty-Q3kaTMsRrfwMXngKVogqEa5ypMCi8?authuser=5#scrollTo=w5zgjCuo84BG&line=1&uniqifier=1 

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))
print(type(cm))
print("Scikit Learn CM: ", cm)

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

names = train_set.classes   # V.V. IMP !!

plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names)