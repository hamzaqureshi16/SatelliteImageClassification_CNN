import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import torch.nn.functional as forward

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=3, stride=1, padding=1)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(in_channels=4096, out_channels=8192, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=8192, out_channels=16384, kernel_size=3, stride=1, padding=1)

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16384, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)

        self.fc4 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.out = nn.Softmax(dim=1)




    def forward(self, x):
        x = forward.relu(self.conv1(x))
        x = forward.relu(self.conv2(x))

        x = self.pool1(x)

        x = forward.relu(self.conv3(x))
        x = forward.relu(self.conv4(x))

        x = self.pool2(x)

        x = forward.relu(self.conv5(x))
        x = forward.relu(self.conv6(x))

        x = self.pool3(x)

        x = forward.relu(self.conv7(x))
        x = forward.relu(self.conv8(x))

        x = self.pool4(x)

        x = forward.relu(self.conv9(x))
        x = forward.relu(self.conv10(x))

        x = self.pool5(x)

        x = x.view(-1, 16384)

        x = forward.relu(self.fc1(x))
        x = self.dropout(x)

        x = forward.relu(self.fc2(x))
        x = self.dropout(x)

        x = forward.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)

        x = self.out(x)

        return x

    def save_model(self, epoch, optimizer, loss, path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }

        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

        return self

