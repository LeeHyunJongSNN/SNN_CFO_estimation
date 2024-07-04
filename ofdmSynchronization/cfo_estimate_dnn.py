import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gc

import torch
import torch.nn as nn

# set up processor use
gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(os.cpu_count() - 1)

# define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.fc4(x)

        return x

net = Net().to(device)

net.load_state_dict(torch.load("models/cfo_dnn_wireless.pth", map_location=device))
net.eval()

def output_cfo(stf):
    return net(stf).detach().numpy().item()

result = output_cfo(torch.tensor(mat_input).float())
