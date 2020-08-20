import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)

    def forward(self, t):

        # (1) input layer
        t = t # Just for understanding, indentity function

        # (2) First Hidden layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        # (3) Second Hidden layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        # (4) 1st Hidden Linear layer

        t = t.reshape(-1,12*4*4)   # Need to flatten the tensor [12 = output shape from prev conv layer] 
                                   # 4,4 = Height, Width of the tensor that is reduced from 28*28 due to conv operations
        t = self.fc1(t)
        t = F.relu(t)

        # (5) 2nd Hidden Linear layer

        t = self.fc2(t)
        t = F.relu(t)

        # (6) Final output layer

        t = self.out(t)
        # t = F.Softmax(t,dim = 1) # No need to do this, PyTorch by default implements cross entropy loss function for us

        return t


torch.set_grad_enabled(False)

network = Network()

print(network)

train_set = torchvision.datasets.FashionMNIST(
    root = '/Users/rushirajsinhparmar/Downloads/PyTorch/',
    train = True,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()])
)

sample = next(iter(train_set))

image, label = sample

output = network(image.unsqueeze(0))
print(output)