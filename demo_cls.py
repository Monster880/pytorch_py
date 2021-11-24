import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
# net
cnn = torch.load("model/mnist_model.pkl")

# loss
loss_func = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr =0.01)

loss_test = 0
accuracy = 0
# training
for i, (images, labels) in enumerate(test_loader):

    # eval/test
    outputs = cnn(images)
    _,pred = outputs.max(1)
    accuracy += (pred == labels).sum().item()

accuracy = accuracy / len(test_data)
print(accuracy)

# load


# inference
