import torch
import torch.nn as nn
import torch.nn.functional as F


class Nature_CNN(nn.Module):
    """
    Input shape: 4x84x84 (C,H,W)
    """

    def __init__(self, n_actions, size='normal',depth='normal'):

        Size = {'normal': 1, 'big': 2, 'super': 4, 'small': 0.5}
        Depth = {'normal':0,'deep':1,'very_deep':2}
        n = Size[size]
        self.k = Depth[depth]

        super(Nature_CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, int(32*n), kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(int(32*n), int(64*n),
                               kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(int(64*n), int(64*n),
                               kernel_size=3, stride=1)
        self.fc1 = nn.Linear(int(7*7*64*n), int(512*n))

        for idx in range(self.k):
            exec("self.fc_{} = nn.Linear(int(512*n),int(512*n))".format(idx))

        self.fc2 = nn.Linear(int(512*n), n_actions)

        self.init_weights()

    def init_weights(self):
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

        for idx in range(self.k):
            exec("x = self.fc_{}(x)".format(idx))
            x = F.relu(x)
        
        x = self.fc2(x)
        return x

    def metrics_names(self):
        return ["loss"]


class ACVP(nn.Module):
    def __init__(self, n_actions):
        super(ACVP, self).__init__()

        self.conv1 = nn.Conv2d(4, 64, 6, 2)
        self.conv2 = nn.Conv2d(64, 64, 6, 2, (2, 2))
        self.conv3 = nn.Conv2d(64, 64, 6, 2, (2, 2))

        self.hidden_units = 64 * 10 * 10

        self.fc4 = nn.Linear(self.hidden_units, 1024)
        self.fc_encode = nn.Linear(1024, 2048)
        self.fc_action = nn.Linear(n_actions, 2048)
        self.fc_decode = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, self.hidden_units)

        self.deconv6 = nn.ConvTranspose2d(64, 64, 6, 2, (2, 2))
        self.deconv7 = nn.ConvTranspose2d(64, 64, 6, 2, (2, 2))
        self.deconv8 = nn.ConvTranspose2d(64, 1, 6, 2)

        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform(layer.weight.data)
            nn.init.constant(layer.bias.data, 0)
        nn.init.uniform(self.fc_encode.weight.data, -1, 1)
        nn.init.uniform(self.fc_decode.weight.data, -1, 1)
        nn.init.uniform(self.fc_action.weight.data, -0.1, 0.1)

    def forward(self, state, action):
        x=F.relu(self.conv1(state))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=x.view((-1, self.hidden_units))
        x=F.relu(self.fc4(x))
        x=self.fc_encode(x)
        action=self.fc_action(action)
        x=torch.mul(x, action)
        x=self.fc_decode(x)
        x=F.relu(self.fc5(x))
        x=x.view((-1, 64, 10, 10))
        x=F.relu(self.deconv6(x))
        x=F.relu(self.deconv7(x))
        x=self.deconv8(x)
        return x

    def metrics_names(self):
        return ["loss"]
