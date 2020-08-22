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


train_set = torchvision.datasets.FashionMNIST(
    root = '/Users/rushirajsinhparmar/Downloads/PyTorch/',
    train = True,
    download = False,
    transform = transforms.Compose([transforms.ToTensor()])
)

network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32)



                    ###################### FOR A SINGLE BATCH ############################# 



bacth = next(iter(train_loader))

images,labels = bacth

# Calculating the loss 

preds = network(images)

loss = F.cross_entropy(preds,labels) # Calculate the loss

loss.item()
print(loss.item())

# Calculating the Gradients

print(network.conv1.weight.grad)

loss.backward()

network.conv1.weight.grad.shape

# Update the weights

optimizer = optim.Adam(network.parameters(), lr = 0.01)

loss.item()

get_num_correct(preds,labels)

optimizer.step()

# VERIFY the TRAINING after one batch

preds = network(images) # JUST TO VERIFY
loss = F.cross_entropy(preds,labels)
loss.item()
get_num_correct(preds,labels)





        ######################### FOR A SINGLE EPOCH IN TRAINLOADER #################################

total_loss = 0
total_correct = 0

for batch in train_loader:
    
    images, labels = batch

    preds = network(images)
    loss = F.cross_entropy(preds,labels)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    total_correct += get_num_correct(preds,labels)

print("epoch: ", 0, "loss: ", loss.item(), "total_correct: ", get_num_correct(preds,labels))



        ###################### FOR ALL EPOCHS IN A TRAIN LOADER ###########################



for epochs in range(5):

    total_loss = 0
    total_correct = 0

    for batch in train_loader: 
        image, labels = batch 

        preds = network(images)
        loss = F.cross_entropy(preds,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)

    print("Epoch: ", epochs, "Loss: ", total_loss, "total_correct: ", total_correct)
