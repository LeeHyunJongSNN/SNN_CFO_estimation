import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gc
import numpy as np

import torch
import torch.nn as nn

# set up processor use
gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(os.cpu_count() - 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(8)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = self.batch_norm(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)

        return x

net = Net().to(device)

net.load_state_dict(torch.load("models/cfo_2dcnn.pth", map_location=device))
net.eval()

def output_cfo(stf):
    stf = stf.view(1, 10, 32)
    return net(stf).detach().numpy().item()

result = output_cfo(torch.tensor(mat_input).float())
