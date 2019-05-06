import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = (nn.Linear(10, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 128, 128)
