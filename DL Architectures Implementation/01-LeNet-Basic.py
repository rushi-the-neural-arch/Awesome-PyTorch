import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class LeNet(nn.Module): 
    def __init__(self):
        super(LeNet,self).__init__()

        self.relu = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        # NOTE: paddinf of (2,2) because MNIST input size is 28 X 28 and LeNet expects
        # input of 32 * 32 ( so acc to formula : 2*p --> 2*2 = 4 (28+4=32))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=(0,0))

        self.linear1 = nn.Linear(120,84)
        self.linear2 = nn.Linear(84,10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x)) # batch_size X 120 X 1 X 1  TO---> batch_size X 120
        # Keep the first dimension (batch_size) same and merge the rest dimensions
        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


#model = LeNet()
# x = torch.randn(64,1,32,32)
# print(model(x).shape)

# TODO: Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on:", device)

# HyperParameters

learning_rate = 0.01
batch_size = 64
num_epochs = 2

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
model = LeNet()
model.to(device)

# TODO: Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

# TODO: num_correct function

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
    

# TODO: Train the network

# for epoch in range(num_epochs):
#     for batch_idx, (data, targets) in enumerate(train_loader):
        
#         total_loss = 0
#         total_correct = 0
#         num_samples = 0
        
#         # Get data to cuda if possible
#         data = data.to(device=device)
#         targets = targets.to(device = device)
        
#         # Forward
#         predictions = model(data)
#         loss = criterion(predictions, targets)
        
#         # Backward
#         optimizer.zero_grad()
#         loss.backward()
        
#         # Gradient Descent or Adam step
#         optimizer.step()
        
#         # NOTE: MULTIPLY BATCH_SIZE !!
#         total_loss += loss.item()* batch_size # V.V.IMPORTANT!!
#         total_correct += get_num_correct(predictions, targets)
#         num_samples += predictions.size(0)
        
# print("Epoch: ", epoch, "Loss: ", total_loss, "total_corect: ", total_correct,
#              "accuracy: ", f'{float((total_correct)/(num_samples))*100:.2f}')


# OPTIONAL

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
                        
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/(num_samples)*100:.2f}')
        
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
