import os
import gc
import numpy as np
from scipy.signal import detrend

import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--input_size", type=int, default=160)
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--output_dim", type=int, default=16)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--rate_decay", type=float, default=0.9)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(gpu=True)
args = parser.parse_args()

seed = args.seed
batch_size = args.batch_size
input_size = args.input_size
n_epochs = args.n_epochs
output_dim = args.output_dim
hidden_dim = args.hidden_dim
num_layers = args.num_layers
learning_rate = args.learning_rate
rate_decay = args.rate_decay
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

# load data and change i to j (complex number)
fname_train = "/home/leehyunjong/Wi-Fi_Preambles/stfcfo/wireless/"\
        "WiFi_10MHz_Preambles_wireless_cfo_train.txt"

raw_train = np.loadtxt(fname_train, dtype='str', delimiter='\t')
np.random.shuffle(raw_train)
for i in range(len(raw_train)):
    for j in range(len(raw_train[i])):
        raw_train[i][j] = raw_train[i][j].replace('i', 'j')

raw_train = raw_train.astype(np.complex64)

# removing DC offsets in signals
train_signals = []

for line in raw_train:
    # line_data = line[160-input_size:160]        # static
    input_size = random.randint(1, 5)  # random
    line_data = line[160 - 32 * input_size:160]
    line_label = np.real(line[-1])
    dcr = detrend(line_data - np.mean(line_data))
    if input_size < 5:   # static -> 160, 2 / random -> 5, 64
        dcr = np.concatenate((np.zeros(160-32*input_size).astype(np.complex64), dcr), axis=0)

    real = np.real(dcr).astype(np.float32)
    imag = np.imag(dcr).astype(np.float32)
    whole = np.concatenate((real, imag), axis=0)
    train_signals.append((whole, float(line_label)))

train_x = torch.tensor(np.stack([i[0] for i in train_signals]), device=device)
train_x = train_x.view(-1, 10, 32)
train_y = torch.tensor(np.expand_dims(np.stack([i[1] for i in train_signals]), 1), device=device)

# dataloader
train = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)

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

# define loss and optimizer
net = Net(train_x[0][0].shape[0], hidden_dim, output_dim, num_layers).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=rate_decay)

# train
net.train()
for epoch in range(n_epochs):
    train_batch = iter(train_loader)
    for inputs, labels in train_batch:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = torch.sqrt(loss_fn(net(inputs), labels.float()) + 1e-6)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        scheduler.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

torch.save(net.state_dict(), "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/cfo_lstm_wireless.pth")
