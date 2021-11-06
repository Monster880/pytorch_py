import torch
# torch.where

a = torch.rand(4,4)
b = torch.rand(4,4)
print(a)
print(b)
out = torch.where(a > 0.5,a,b)
print(out)

# torch.index select
print("torch.index select")
a = torch.rand(4,4)
print(a)
out = torch.index_select(a, dim=0, index = torch.tensor([0,3,2]))

print(out,out.shape)

# torch.gather
print("torch.gather")
a = torch.linspace(1, 16, 16).view(4,4)
print(a)
out = torch.gather(a, dim=0, index=torch.tensor([[0, 1, 1, 1],
                                                [0, 1, 2, 2],
                                                [0, 1, 3, 3]]))
print(out, out.shape)

# torch.masked_index
print("masked_index")
a = torch.linspace(1,16,16).view(4,4)
mask = torch.gt(a, 8)
print(a)
print(mask)
out = torch.masked_select(a, mask)
print(out)

# torch.take
print("torch.take")
a = torch.linspace(1,16,16).view(4,4)
b = torch.take(a, index=torch.tensor([0,15,13,10]))
print(b)

# torch.nonzero
print("torch.nonzero")
a = torch.tensor([[0,1,2,0],[2,3,0,1]])
out = torch.nonzero(a)
print(out)
# 稀疏表示
