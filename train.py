import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from load_cifar10 import train_data_loader, test_data_loader
import os

# 是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch = 200
lr = 0.01

net = VGGNet().to(device)

# loss
loss_func = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
# optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
correct = 0
batch_size = 0
if __name__ == '__main__':
    for epoch in range(epoch):
        print("epoch is ", epoch)
        net.train()  # training BN dropout

        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(outputs, dim = 1)
            batch_size += labels.size(0)
            correct += pred.eq(labels.data).cpu().sum()
            print("step", i, "loss is:", loss.item(), "mini-batch correct is:", 100.0 * correct / batch_size)

        if os.path.exists("models"):
            os.mkdir("models")
        torch.save(net.state_dict(), "models/{}.pth", format(epoch + 1))
        scheduler.step()

        for i, data in enumerate(test_data_loader):
            net.eval()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()