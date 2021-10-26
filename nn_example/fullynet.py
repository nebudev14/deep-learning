import torch
import torch.nn as nn # nn modules
import torch.optim as optim # optimization functions
import torch.nn.functional as F # funcs with no params (relu, etc)
from torch.utils.data import DataLoader # easier dataset management
import torchvision.datasets as datasets # standard datasets to test (mnist dataeset)
import torchvision.transforms as transforms # transformations to peform on dataset 

# create fully connected network
class NN(nn.Module): # class neural network inherits nn.module
    def __init__(self, input_size, num_classes): # (28 * 28 = 784) input size
        super(NN, self).__init__() # initialize parent class nn.module
        self.fc1 = nn.Linear(input_size, 50) # layer 1, 50 nodes
        self.fc2 = nn.Linear(50, num_classes) # layer 2, 50 nodes to the num of classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True) # training dataset, transforms to tensor in pytorch
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True) 
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# init network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(num_epochs): # 1 epoch = network has seen all the images in the dataset
    for batch_idx, (data, targets) in enumerate(train_loader): # data = image, target = correct number
        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # get to correct shape
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward (backprop)
        optimizer.zero_grad() # set all gradients to zero for each batch so it doesnt have backprop calcs from previous forward props
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        
# check accuacy on training and test to see hwo good our model is

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            scores.max()

