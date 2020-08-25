import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)

    def forward(self,t):

        # (1) Input Layer
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
        t = t.flatten(start_dim = 1)  # t = t.reshape(-1, 12*4*4)
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


network = Network()

######################## Creating RUN_BUILDER_CLASS #####################################


from itertools import product

from collections import namedtuple
from collections import OrderedDict


parameters = OrderedDict(
    lr = [0.01, 0.05, 0.1],
    batch_size = [10,32,64],
)

class RunBuilder():

    @staticmethod
    def get_runs(params): 

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs 

runs = RunBuilder.get_runs(parameters)

print(runs)

run = runs[0]
print(run)

print(run.lr, run.batch_size)

for run in runs:
    print(run.lr, run.batch_size)


for run in RunBuilder.get_runs(parameters):
    print(run.lr, run.batch_size)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = run.batch_size)
    optimizer = optim.Adam(network.parameters(), lr = run.lr)

    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)

    comment = f'-{run}'

    tb = SummaryWriter(comment = comment)
    tb.add_image('images', grid)
    tb.add_graph(network, images)

    for epoch in range(2):

        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            images, labels = batch

            preds = network(images)
            loss = F.cross_entropy(preds,labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # NOTE: MULTIPLY BATCH_SIZE !!
            total_loss += loss.item() * run.batch_size              ## V.V.IMPORTANT!!!!
            total_correct += get_num_correct(preds,labels)

        tb.add_scalar("Loss", total_loss, epoch)
        tb.add_scalar("total_correct", total_correct, epoch)
        tb.add_scalar("Accuracy", total_correct/len(train_set), epoch)

        for name, weight in network.named_parameters():
            tb.add_histogram(name,weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)


        print("Epoch: ", epoch, "Loss: ", total_loss, "total_corect: ", total_correct)


tb.close()