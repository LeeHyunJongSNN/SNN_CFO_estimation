import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gc

import torch
import torch.nn as nn

import numpy as np

# set up processor use
gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(os.cpu_count() - 1)
batch_size = 100
delta_freq = 49680

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(2, 128, 7, padding=3)
        self.conv2 = nn.Conv1d(128, 128, 5, padding=3)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(128, 128, 7, padding=3)
        self.conv4 = nn.Conv1d(128, 128, 5, padding=3)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv5 = nn.Conv1d(128, 128, 7, padding=3)
        self.conv6 = nn.Conv1d(128, 128, 5, padding=3)
        self.pool3 = nn.MaxPool1d(2, 2)
        self.conv7 = nn.Conv1d(128, 128, 7, padding=3)
        self.conv8 = nn.Conv1d(128, 128, 5, padding=3)
        self.pool4 = nn.MaxPool1d(2, 2)
        self.conv9 = nn.Conv1d(128, 128, 7, padding=3)
        self.conv10 = nn.Conv1d(128, 128, 5, padding=3)
        self.pool5 = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.BatchNorm1d(128, device=device)(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.BatchNorm1d(128, device=device)(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = nn.BatchNorm1d(128, device=device)(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.BatchNorm1d(128, device=device)(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = nn.BatchNorm1d(128, device=device)(x)
        x = nn.functional.relu(x)
        x = self.conv6(x)
        x = nn.BatchNorm1d(128, device=device)(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = self.conv7(x)
        x = nn.BatchNorm1d(128, device=device)(x)
        x = nn.functional.relu(x)
        x = self.conv8(x)
        x = nn.BatchNorm1d(128, device=device)(x)
        x = nn.functional.relu(x)
        x = self.pool4(x)
        x = self.conv9(x)
        x = nn.BatchNorm1d(128, device=device)(x)
        x = nn.functional.relu(x)
        x = self.conv10(x)
        x = nn.BatchNorm1d(128, device=device)(x)
        x = nn.functional.relu(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.tanh(x)

        return x

net = Net().to(device)

net.load_state_dict(torch.load("models/cfo_prontol_wireless.pth", map_location=device))
net.eval()

def output_cfo(ltf):
    ltf = ltf.view(1, 2, 160)
    return net(ltf).detach().numpy().item()

input_real = np.real(mat_input).astype(np.float32)
input_imag = np.imag(mat_input).astype(np.float32)
real_rms = np.sqrt(np.sum(np.power(np.abs(input_real), 2)) / 160)
imag_rms = np.sqrt(np.sum(np.power(np.abs(input_imag), 2)) / 160)
net_input = np.stack((input_real / real_rms, input_imag / imag_rms))

result = output_cfo(torch.tensor(net_input).float()) * delta_freq
