import torch
import torch.nn as nn # nn modules
import torch.optim as optim # optimization functions
import torch.nn.functional as F # funcs with no params (relu, etc)
from torch.utils.data import DataLoader # easier dataset management
import torchvision.datasets as datasets # standard datasets to test (mnist dataeset)
import torchvision.transforms as transforms # transformations to peform on dataset 