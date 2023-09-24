import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self) :
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1,padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
        # 32channels 7*7 mat
        
    def forward(self, x):
        x = self.conv1(x)
        # 1st conv layer
        x = self.relu(x)
        # activation func. (only once before pooling)
        x = self.pool(x)
        x = self.conv2(x)
        # after pooling, no more activation func.
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x