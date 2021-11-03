import torch

a = torch.rand(2,4,1,3)
b = torch.rand(4,2,3)
c = a + b

# 2*4*2*3
print(a)
print(b)
print(c)
print(c.shape)
