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