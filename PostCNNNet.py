#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch.nn as nn

class PostCNNNet(nn.Module):
    def __init__(self):
        super(PostCNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25))
        self.bn1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25))
        self.bn2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        self.fc1 = nn.Linear(32 * 45, 64)  # Flattened output from conv layers
        self.fc2 = nn.Linear(64, 4)  # Output layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = torch.relu(self.fc1(x))  # Fully connected layer
        x = self.fc2(x)  # Output layer
        return x


# In[ ]:




