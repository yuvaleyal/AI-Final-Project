import torch
import torch.nn as nn
from Constants import *
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    def __init__(self):
        super(PolicyValueNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Input channels: 4 (current player's pieces and opponent's pieces)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, ACTION_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value
