import torch
from torch import nn
from torch.nn import functional as F
import copy


class DigitNet(nn.Module):
    def __init__(self, nb_hidden):
        super(DigitNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        # no need for softmax here because crossEntropyLoss() already applies one
        return x


class CompNet(torch.nn.Module):
    def __init__(self, digitnet_1, digitnet_2=None, weight_sharing=True):
        super(CompNet, self).__init__()

        self.weight_sharing = weight_sharing

        if self.weight_sharing:
            self.digitNet = digitnet_1
        else:
            self.digitNet1 = digitnet_1
            self.digitNet2 = digitnet_2

        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x1, x2, train=True):
        if self.weight_sharing:
            x1 = self.digitNet.forward(x1)
            x2 = self.digitNet.forward(x2)
        else:
            x1 = self.digitNet1.forward(x1)
            x2 = self.digitNet2.forward(x2)
        x = torch.cat((x1, x2), 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=train)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(x)
        return x
