import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HEIGHT, WIDTH, lstm_seq_length

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.head(x)


class DQN_LSTM(nn.Module):
    def __init__(self, action_size):
        super(DQN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(256, action_size)
        # Define an LSTM layer

    def forward(self, x, hidden = None, train=True):
        if train==True: # If training, we will merge all the visual states into one. So first two dimensions (batch_size, lstm_seq_length) will be merged into one. This will let us process all these together.
            x = x.view(-1, 1, HEIGHT, WIDTH)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        if train==True: # We will reshape the output to match the original shape. So first dimension will be extended back to (batch_size, lstm_seq_length)
            x = x.view(-1, lstm_seq_length, 512)
        # Pass the state through an LSTM
        ### CODE ###

        return self.head(lstm_output), hidden