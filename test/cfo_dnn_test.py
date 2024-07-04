import os
import gc
import numpy as np
from scipy.signal import detrend

import argparse

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(plot=False, gpu=True)
args = parser.parse_args()

seed = args.seed
batch_size = args.batch_size
gpu = args.gpu
spare_gpu = args.spare_gpu

# set up gpu use
gc.collect()
torch.cuda.empty_cache()

if spare_gpu != 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spare_gpu)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)

fname_test = "/home/leehyunjong/Wi-Fi_Preambles/stfcfo/wireless/"\
        "WiFi_10MHz_Preambles_wireless_cfo_test_rician_-3dB.txt"

raw_test = np.loadtxt(fname_test, dtype='str', delimiter='\t')
np.random.shuffle(raw_test)
for i in range(len(raw_test)):
    for j in range(len(raw_test[i])):
        raw_test[i][j] = raw_test[i][j].replace('i', 'j')

raw_test = raw_test.astype(np.complex64)

# removing DC offsets in signals
test_signals = []

for line in raw_test:
    line_data = line[0:160]
    line_label = np.real(line[-1])
    dcr = detrend(line_data - np.mean(line_data))
    real = np.real(dcr).astype(np.float32)
    imag = np.imag(dcr).astype(np.float32)
    whole = np.concatenate((real, imag), axis=0)
    test_signals.append((whole, float(line_label)))

test_x = torch.tensor(np.stack([i[0] for i in test_signals]), device=device)
test_y = torch.tensor(np.expand_dims(np.stack([i[1] for i in test_signals]), 1), device=device)

# dataloader
test = torch.utils.data.TensorDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

# define measurement
def MAE(y, y_hat):
    return np.mean(np.abs(y - y_hat))

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

# load model
net = Net().to(device)

# test
net.load_state_dict(torch.load("/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/cfo_dnn_wireless.pth"))
net.eval()
test_outputs = []
test_labels = []
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    test_outputs.append(net(inputs).cpu().detach().numpy())
    test_labels.append(labels.cpu().detach().numpy())

# measurements
test_outputs = np.array(test_outputs).squeeze().reshape(1, -1).squeeze()
test_labels = np.array(test_labels).squeeze().reshape(1, -1).squeeze()
mae = MAE(test_outputs, test_labels)
print(f"MAE: {mae.item()}")
