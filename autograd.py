import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y*y*2
z = z.mean()
print(z)

z.backward() # dz/dx
print(x.grad)
print("-------")

x = torch.randn(4, requires_grad=True)
y = x+2
z = y*y*2
print(z)

v = torch.tensor([0.1, 1.0, 0.01, 0.02])
z.backward(v)
print(x.grad)
