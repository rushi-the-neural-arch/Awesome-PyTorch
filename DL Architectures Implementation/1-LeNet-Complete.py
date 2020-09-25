import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from datetime import datetime

# TODO - 1: check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO - 2: PARAMETERS
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 5

IMG_SIZE = 32
N_CLASSES = 10

# TODO - 3: Data

# define transforms
transforms = transforms.Compose([transforms.Resize((32,32)),
                                 transforms.ToTensor()])          

# download and create datasets
train_dataset = datasets.FashionMNIST(root='/home/rushirajsinh/repos/DeepLizard-PyTorch/',
                                      train = True,
                                      transform=transforms,
                                      download = False)

valid_dataset = datasets.FashionMNIST(root='/home/rushirajsinh/repos/DeepLizard-PyTorch/',
                                      train = False,
                                      transform=transforms)


# define the dataloader
train_loader = DataLoader(dataset = train_dataset,
                          batch_size= BATCH_SIZE,shuffle=True)

valid_loader = DataLoader(dataset = valid_dataset,
                          batch_size= BATCH_SIZE,shuffle=False)

# Plotting the images

ROW_IMG = 10
N_ROWS = 5

fig = plt.figure()
for index in range(1, ROW_IMG * N_ROWS + 1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis('off')
    plt.imshow(train_dataset.data[index])
fig.suptitle('MNIST Dataset preview'); 


fig = plt.figure()
for index in range(1, ROW_IMG * N_ROWS + 1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis('off')
    plt.imshow(train_dataset.data[index], cmap='gray_r')
fig.suptitle('MNIST Dataset - preview'); 


# TODO - 4: Helper Functions:

def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()

        for X,y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)
             
            _, y_prob = model(X) # model(x) returns a tensor of size (BATCH_SIZE, NUM_CLASSES) - tensor([32,10])
            _, predicted_labels = torch.max(y_prob, dim=1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    # temporarily change the style of plots to seaborn
    plt.style.use('seaborn')

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8,4.5))

    ax.plot(train_losses, color='blue', label = 'Training Loss')
    ax.plot(valid_losses, color='red', label = 'Validation Loss')
    
    ax.set(title = 'Loss over epochs', xlabel='Epoch', ylabel='Loss')
    ax.legend()
    fig.show()

    # change the plot style to default
    plt.style.use('default')

# TODO 5: Training Helper Functions

def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of training loop
    
    '''
    model.train()
    running_loss = 0

    for X,y_true in train_loader:

        X = X.to(device)
        y_true = y_true.to(device)

        # print("X.size(0)=", X.size(0)) - 32 (BATCH_SIZE)
        # Forward pass       
        preds, _ = model(X)
        loss = criterion(preds, y_true)
        running_loss += loss.item()*X.size(0) 

        # print("Loss.item() = ", loss.item()) - Prints out loss at every batch
        # print("RUNNING LOSS \n:", running_loss)
        
        '''
        V.V.IMP: https://stackoverflow.com/questions/61092523/what-is-running-loss-in-pytorch-and-how-is-it-calculated#:~:text=item()%20contains%20the%20loss,batch%20size%2C%20given%20by%20inputs.
        Check the abve link for reference understanding

        # if the batch_size is 4, loss.item() would give the loss for the entire set of 4 images
        That depends on how the loss is calculated. Remember, loss is a tensor just like every other tensor. 
        In general the PyTorch APIs return avg loss by default
        "The losses are averaged across observations for each minibatch."
        '''

        # Backward pass
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

    # print(len(train_loader.dataset)) - 60,000

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, optimizer, device):
    ''' 
    Function for validation step of training loop
    '''

    model.eval()
    running_loss = 0

    for X,y_true in valid_loader:

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        preds, _ = model(X)
        loss = criterion(preds, y_true) # Computes the loss eg- CrossEntropy
        running_loss += loss.item()*X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss



def training_loop(model, criterion, optimizer,train_loader, valid_loader,
                 epochs, device, print_every = 1):

    '''
    Function defining the entire training loop
    '''
    # set objects for storinh metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, optimizer, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == 0:

            train_acc = get_accuracy(model, train_loader, device = device)
            valid_acc = get_accuracy(model, valid_loader, device = device)

            print(f'{datetime.now().time().replace(microsecond=0)}---'
                  f'Epoch: {epoch}\t'
                  f'Train Loss: {train_loss:.4f}\t'
                  f'Valid Loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}\t')

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)



# TODO: 6 -  IMPLEMENT LE-NET Architecture

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16,kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
            )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
            )

    def forward(self, x):

        x = self.feature_extractor(x)
        x = x.reshape(x.shape[0],-1)

        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)

        return logits, probs


# TODO 7: Start Training

torch.manual_seed(RANDOM_SEED)

model = LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, 
                                    valid_loader, N_EPOCHS, DEVICE)