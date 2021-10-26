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
    

# model = NN(784, 10) # 784 inputs, 10 digits
# x = torch.rand(64, 784) # num of images, input size
# print(model(x).shape)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# load data