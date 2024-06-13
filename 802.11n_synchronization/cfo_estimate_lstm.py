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

# define model
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_cells = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_cells, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_cells, x.size(0), self.hidden_dim).to(device)

        x, h_s = self.lstm(x, (h0, c0))
        x = self.fc1(x[:, -1, :])
        x = nn.functional.relu(x)
        x = self.fc2(x)

        return x

net = Net(32, 256, 16, 1).to(device)

net.load_state_dict(torch.load("models/cfo_lstm.pth", map_location=device))
net.eval()

def output_cfo(stf):
    stf = stf.view(1, 10, 32)
    return net(stf).detach().numpy().item()

result = output_cfo(torch.tensor(mat_input).float())

