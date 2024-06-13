import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gc
import numpy as np

import torch
import torch.nn as nn

from spikingjelly.activation_based import surrogate, neuron, functional

# set up processor use
gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(os.cpu_count() - 1)
batch_size = 100

# define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640, 512)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)

        r = 0
        n_steps = 4
        for i in range(n_steps):
            nif = self.fc1(self.flatten(x))
            nif = self.if1(nif)
            r += nif

        s = r / n_steps
        x = self.fc2(s)
        x = nn.functional.relu(x)
        x = self.fc3(x)

        return x

net = Net().to(device)

net.load_state_dict(torch.load("models/cfo_scnn.pth", map_location=device))
net.eval()

def output_cfo(stf):
    stf = stf.view(1, 10, 32)
    return net(stf).detach().numpy().item()

result = output_cfo(torch.tensor(mat_input).float())
