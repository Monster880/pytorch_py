import torch;

a = torch.Tensor([[1,2], [3,4]])
print(a)
print(a.type())

a = torch.Tensor(2,3)
print(a)
print(a.type())

a = torch.ones(2,2)
print(a)
print(a.type())

a = torch.zeros(2,2)
print(a)
print(a.type())

b = torch.Tensor(2,3)
b = torch.zeros_like(b)
print(b)
print(b.type())