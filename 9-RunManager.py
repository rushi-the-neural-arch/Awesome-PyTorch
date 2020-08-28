import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

class Network(nn.Module):
  def __init__(self):
    super(Network,self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size = 5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size = 5)

    self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=10)

  def forward(self,t):

    # (1) Input Layer
    t = t

    # (2) First Hidden Conv Layer
    t = self.conv1(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # (3) Second Hidden Conv Layer
    t = self.conv2(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # (4) 1st Hidden Linear Layer
    t = t.flatten(start_dim = 1)   # t = t.reshape(-1, 12*4*4)
    t = self.fc1(t)
    t = F.relu(t)

    # (5) 2nd Hidden Linear Layer
    t = self.fc2(t)
    t = F.relu(t)

    # (6) 3rd Hidden Linear Layer
    t = self.out(t)

    return t


train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMnist',
    train = True,
    download = False,
    transform = transforms.Compose([transforms.ToTensor()])
)

network = Network()

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import time
import json

from itertools import product

import time
from collections import OrderedDict
from collections import namedtuple

class RunBuilder():

  @staticmethod
  def get_runs(params):

    Run = namedtuple('Run', params.keys())

    runs = []
    for v in product(*params.values()):
      runs.append(Run(*v))

    return runs

parameters = OrderedDict(
    lr = [0.01],
    batch_size = [1000,2000],
)

runs = RunBuilder.get_runs(parameters)
print(runs)


class RunManager():
  def __init__(self):

    self.epoch_count = 0
    self.epoch_loss = 0
    self.epoch_num_correct = 0
    self.epoch_start_time = None

    self.run_params = None
    self.run_count = 0
    self.run_data = []
    self.run_start_time = None

    self.network = None
    self.loader = None
    self.tb = None 

  def begin_run(self, run, network, loader):

    self.run_start_time = time.time()
    self.run_count += 1

    self.network = network
    self.loader = loader
    self.tb = SummaryWriter(comment = f'-{run}')

    images, labels = next(iter(self.loader))
    grid = torchvision.utils.make_grid(images)

    self.tb.add_image('images', grid)
    self.tb.add_graph(self.network, images)

  def end_run(self):

    self.tb.close()
    self.epoch_count = 0

  def begin_epoch(self):

    self.epoch_start_time = time.time()
    self.epoch_count += 1
    self.epoch_loss = 0
    self.epoch_num_correct = 0

  def end_epoch(self):

    epoch_duration = time.time() - self.epoch_start_time
    run_duration = time.time() - self.run_start_time

    loss = self.epoch_loss / len(self.loader.dataset)
    accuracy = self.epoch_num_correct / len(self.loader.dataset)

    self.tb.add_scalar("Loss", loss, self.epoch_count)
    self.tb.add_scalar("Accuracy", accuracy, self.epoch_count)

    for name, param in self.network.named_parameters():
      self.tb.add_histogram(name, param, self.epoch_count)
      self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

    results = OrderedDict()

    results["run"] = self.run_count
    results["epoch"] = self.epoch_count
    results["loss"] = loss
    results["accuracy"] = accuracy
    results["epoch duration"] = epoch_duration
    results["run_duration"] = run_duration

    for k,v in self.run_params._asdict().items():
      results[k] = v 

    self.run_data.append(results)

    df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')

    clear_output(wait = True)
    display(df)

  
  def get_number_correct(self, preds, labels):
      return preds.argmax(dim = 1).eq(labels).sum().item()  

  def track_loss(self,loss):
      self.epoch_loss += loss.item()*self.loader.batch_size

  def track_num_correct(self, preds, labels):
      self.epoch_num_correct += self.get_number_correct(preds, labels)


  def save(save, filename):

      pd.DataFrame.from_dict(
          self.run_data,
          orient = 'columns'
      ).to_csv(f'{filename}.csv')

      with open(f'{filename}.json', 'w', encoding = 'utf-8') as f:
        json.dump(self.run_data, f, ensure_ascii = False, indent = 4)

network = Network()

m = RunManager()

for run in RunBuilder.get_runs(parameters):

  loader = DataLoader(train_set, batch_size = run.batch_size)
  optimizer = optim.Adam(network.parameters(), lr = 0.01)

  m.begin_run(run, network, loader)

  for epoch in range(5):
    m.begin_epoch()

    for batch in loader:

        images, labels = batch
        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        m.track_loss(loss)
        m.track_num_correct(preds,labels)

    m.end_epoch()
  m.end_run()

m.save('results')