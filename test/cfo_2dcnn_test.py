import os
import gc
import numpy as np
from scipy.signal import detrend

import argparse
import random

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--input_size", type=int, default=64)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(gpu=True)
args = parser.parse_args()

seed = args.seed
batch_size = args.batch_size
input_size = args.input_size
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
        "WiFi_10MHz_Preambles_wireless_cfo_test_rician_0dB.txt"

raw_test = np.loadtxt(fname_test, dtype='str', delimiter='\t')
np.random.shuffle(raw_test)
for i in range(len(raw_test)):
    for j in range(len(raw_test[i])):
        raw_test[i][j] = raw_test[i][j].replace('i', 'j')

raw_test = raw_test.astype(np.complex64)

# removing DC offsets in signals
test_signals = []

for line in raw_test:
    line_data = line[160 - input_size:160]
    line_label = np.real(line[-1])
    dcr = detrend(line_data - np.mean(line_data))
    if input_size < 160:
        dcr = np.concatenate((np.complex64(np.zeros(160 - input_size)), dcr), axis=0)

    real = np.real(dcr).astype(np.float32)
    imag = np.imag(dcr).astype(np.float32)
    whole = np.concatenate((real, imag), axis=0)
    test_signals.append((whole, float(line_label)))

test_x = torch.tensor(np.stack([i[0] for i in test_signals]), device=device)
test_x = test_x.view(5000, -1, 32)
test_y = torch.tensor(np.expand_dims(np.stack([i[1] for i in test_signals]), 1), device=device)

# data loader
test = torch.utils.data.TensorDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

# define measurement
def MAE(y, y_hat):
    return np.mean(np.abs(y - y_hat))

# define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
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

# load model
net = Net().to(device)

# test
net.load_state_dict(torch.load("/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/cfo_2dcnn_wireless.pth"))
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

test_outputs_1 = test_outputs[0:500]
test_labels_1 = test_labels[0:500]
test_outputs_2 = test_outputs[500:1000]
test_labels_2 = test_labels[500:1000]
test_outputs_3 = test_outputs[1000:1500]
test_labels_3 = test_labels[1000:1500]
test_outputs_4 = test_outputs[1500:2000]
test_labels_4 = test_labels[1500:2000]
test_outputs_5 = test_outputs[2000:2500]
test_labels_5 = test_labels[2000:2500]
test_outputs_6 = test_outputs[2500:3000]
test_labels_6 = test_labels[2500:3000]
test_outputs_7 = test_outputs[3000:3500]
test_labels_7 = test_labels[3000:3500]
test_outputs_8 = test_outputs[3500:4000]
test_labels_8 = test_labels[3500:4000]
test_outputs_9 = test_outputs[4000:4500]
test_labels_9 = test_labels[4000:4500]
test_outputs_10 = test_outputs[4500:5000]
test_labels_10 = test_labels[4500:5000]

mae_1 = MAE(test_outputs_1, test_labels_1)
mae_2 = MAE(test_outputs_2, test_labels_2)
mae_3 = MAE(test_outputs_3, test_labels_3)
mae_4 = MAE(test_outputs_4, test_labels_4)
mae_5 = MAE(test_outputs_5, test_labels_5)
mae_6 = MAE(test_outputs_6, test_labels_6)
mae_7 = MAE(test_outputs_7, test_labels_7)
mae_8 = MAE(test_outputs_8, test_labels_8)
mae_9 = MAE(test_outputs_9, test_labels_9)
mae_10 = MAE(test_outputs_10, test_labels_10)
mae = MAE(test_outputs, test_labels)

print(f"MAE: {mae_1.item()}")
print(f"MAE: {mae_2.item()}")
print(f"MAE: {mae_3.item()}")
print(f"MAE: {mae_4.item()}")
print(f"MAE: {mae_5.item()}")
print(f"MAE: {mae_6.item()}")
print(f"MAE: {mae_7.item()}")
print(f"MAE: {mae_8.item()}")
print(f"MAE: {mae_9.item()}")
print(f"MAE: {mae_10.item()}")
print(f"Average MAE: {mae.item()}")
