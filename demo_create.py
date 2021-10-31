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

# 随机
a = torch.rand(2,2)
print(a)
print(a.type())

a = torch.normal(mean=0.0, std=torch.rand(5))
print(a)
print(a.type())

a = torch.normal(mean=torch.rand(5), std=torch.rand(5))
print(a)
print(a.type())

torch.Tensor(2,2).uniform_(-1, 1)
print(a)
print(a.type())

# 序列
a = torch.arange(0, 10, 1)
print(a)
print(a.type())

a = torch.linspace(2, 10, 4)    # 拿到等间隔的n个数字
print(a)
print(a.type())

a = torch.randperm(10)
print(a)
print(a.type())

####
import numpy as np
a = np.array([[1,2],[2,3]])
print(a)