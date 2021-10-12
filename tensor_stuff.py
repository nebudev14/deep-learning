import torch
import numpy as np

x = torch.empty(2, 3, 2, 2) # empty tensor

x = torch.rand(2, 2) # random values

x = torch.ones(3, 2, dtype=torch.int) # tensors full of ones

x = torch.tensor([2.5, 0.1], dtype=torch.double)

print(x)
print(x.dtype)
print(x.size())
print("------")

# adding tensors together
x = torch.rand(2, 2)
y = torch.rand(2, 2)
y *= x
z = x + y
z = torch.add(x, y)
print(z)

print("------")
x = torch.tensor([2, 3])
y = torch.tensor([4, 5])
print(x*y)
print("------")

x = torch.rand(3, 5)
print(x)
print(x[:, 0]) # get every row in the first column
print(x[1, 2].item()) # get actual exact value

# converting from numpy arrays to tensors
print("------")
a = torch.ones(5)
print(a)
print(type(a))
b = a.numpy()
print(b)
print(type(b))

a.add_(1)
print(a, b) # both point to same memory location, changing one will change the other

# 26:00