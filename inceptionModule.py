import torch
import torch.nn as nn

# input:A
# resnet: B = g(A) + f(A)
# Inception:
# B1 = f1(A)
# B2 = f2(A)
# B3 = f3(A)
#
# concat([B1, B2, B3])

def ConvBNRelu(in_channel, out_channel, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=1)
    )

class BaseInception(nn.Module):
    def __init__(self):
        super(BaseInception, self).__init__()

class InceptionNet(nn.Module):
    def __init__(self):
        raise

    def forward(self):
        raise