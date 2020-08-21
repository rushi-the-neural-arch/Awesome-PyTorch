import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True) # True by default

print(torch.__version__)
print(torchvision.__version__)

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12,kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 12*4*4,out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 6, out_features = 10)

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

        # (4) 1st Hidden Linear layer
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) 2nd Hidden Linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) Output layer
        t = self.out(t)

        return t