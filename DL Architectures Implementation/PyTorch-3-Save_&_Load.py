# Imports
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

# TODO: Create a simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN, self).__init__()
        
        # same convolution - OUTPUT size will change - Use VALID if you dont want to change
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 8,kernel_size = (3,3),stride = (1,1),padding = (1,1))
        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        # We will use this same poolong layer everywhere
        
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes) 
        # 7 - because we will use the pooling layer twice - from 28 to 14 to 7
        
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        
        return x

# TODO: Create a CheckPoint functional

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# TODO: Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on:", device)

# HyperParameters

learning_rate = 0.01
batch_size = 64
num_epochs = 6
load_model = True 




# TODO: Load Data

train_dataset = datasets.FashionMNIST(
    root = '/Users/rushirajsinhparmar/Downloads/PyTorch/', train = True, transform = transforms.ToTensor(),
    download = False) # Keep download=True if you dont have the dataset

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = datasets.FashionMNIST(
    root = '/Users/rushirajsinhparmar/Downloads/PyTorch/', train = False, transform = transforms.ToTensor(),
    download = False) # Keep download=True if you dont have the dataset

test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)




# TODO: Initialise the Network

model = CNN().to(device)



# TODO: Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))



# TODO: num_correct function

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
    




# TODO: Train the network

for epoch in range(num_epochs):

    if epoch % 2 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        

        total_loss = 0
        total_correct = 0
        num_samples = 0
        
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device = device)
        
        # Forward
        predictions = model(data)
        loss = criterion(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Descent or Adam step
        optimizer.step()
        
        # NOTE: MULTIPLY BATCH_SIZE !!
        total_loss += loss.item()* batch_size # V.V.IMPORTANT!!
        total_correct += get_num_correct(predictions, targets)
        num_samples += predictions.size(0)
        
    print("Epoch: ", epoch, "Loss: ", total_loss, "total_corect: ", total_correct,
             "accuracy: ", f'{float((total_correct)/(num_samples))*100:.2f}')


# OPTIONAL

# def check_accuracy(loader, model):
#     if loader.dataset.train:
#         print("Checking accuracy on training data")
#     else:
#         print("Checking accuracy on testing data")
    
#     num_correct = 0
#     num_samples = 0
    
#     model.eval()
    
#     with torch.no_grad():
#         for x,y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)
                        
#             scores = model(x)
#             _, predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)
            
#         print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/(num_samples)*100:.2f}')
        
#     model.train()

# check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)
