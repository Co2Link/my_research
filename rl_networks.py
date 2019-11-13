import torch
import torch.nn as nn
import torch.nn.functional as F


class Nature_CNN(nn.Module):
    """
    Input shape: 4x84x84 (C,H,W)
    """

    def __init__(self, n_actions, size='normal'):

        Size = {'normal': 1, 'big': 2, 'super': 4, 'small': 0.5}
        n = int(Size[size])

        super(Nature_CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32*n, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32*n, 64*n,
                               kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64*n, 64*n,
                               kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*64*n, 512*n)
        self.fc2 = nn.Linear(512*n, n_actions)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def metrics_names(self):
        return ["loss"]
