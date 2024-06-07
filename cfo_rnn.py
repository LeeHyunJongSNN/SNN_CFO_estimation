import os
import gc
import numpy as np
from scipy.signal import detrend
from sklearn.preprocessing import MinMaxScaler

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--output_dim", type=int, default=16)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(plot=False, gpu=True)
args = parser.parse_args()

seed = args.seed
batch_size = args.batch_size
n_epochs = args.n_epochs
output_dim = args.output_dim
hidden_dim = args.hidden_dim
num_layers = args.num_layers
learning_rate = args.learning_rate
plot = args.plot
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
        "WiFi_10MHz_Preambles_wireless_cfo_rician_alldB.txt"

fname_test = "/home/leehyunjong/Wi-Fi_Preambles/stfcfo/wireless/"\
        "WiFi_10MHz_Preambles_wireless_cfo_rician_-3dB.txt"

raw_train = np.loadtxt(fname_train, dtype='str', delimiter='\t')
np.random.shuffle(raw_train)
for i in range(len(raw_train)):
    for j in range(len(raw_train[i])):
        raw_train[i][j] = raw_train[i][j].replace('i', 'j')

raw_train = raw_train.astype(np.complex64)

raw_test = np.loadtxt(fname_test, dtype='str', delimiter='\t')
np.random.shuffle(raw_test)
for i in range(len(raw_test)):
    for j in range(len(raw_test[i])):
        raw_test[i][j] = raw_test[i][j].replace('i', 'j')

raw_test = raw_test.astype(np.complex64)

# removing DC offsets in signals
train_signals = []
test_signals = []

for line in raw_train:
    line_data = line[0:160]
    line_label = line[-1]
    dcr = detrend(line_data - np.mean(line_data))
    # fft = np.array([])
    # for i in range(0, 160, 16):
    #     fft = np.concatenate((fft, np.fft.fft(dcr[i:i+16]) / 16), axis=0)
    imag = np.imag(dcr).astype(np.float32)
    real = np.real(dcr).astype(np.float32)
    whole = np.concatenate((real, imag), axis=0)

    train_signals.append((whole, float(line_label)))

for line in raw_test:
    line_data = line[0:160]
    line_label = line[-1]
    dcr = detrend(line_data - np.mean(line_data))
    # fft = np.array([])
    # for i in range(0, 160, 16):
    #     fft = np.concatenate((fft, np.fft.fft(dcr[i:i+16]) / 16), axis=0)
    imag = np.imag(dcr).astype(np.float32)
    real = np.real(dcr).astype(np.float32)
    whole = np.concatenate((real, imag), axis=0)

    test_signals.append((whole, float(line_label)))

# scaling and splitting data into train and test
scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
train_scaled_x = scaler_x.fit_transform(np.stack([i[0] for i in train_signals]))
train_scaled_y = scaler_y.fit_transform(np.expand_dims(np.stack([i[1] for i in train_signals]), 1))
test_scaled_x = scaler_x.transform(np.stack([i[0] for i in test_signals]))
test_scaled_y = scaler_y.transform(np.expand_dims(np.stack([i[1] for i in test_signals]), 1))

train_x = torch.tensor(train_scaled_x, device=device)
train_y = torch.tensor(train_scaled_y, device=device)
test_x = torch.tensor(test_scaled_x, device=device)
test_y = torch.tensor(test_scaled_y, device=device)

# dataloader
train = torch.utils.data.TensorDataset(train_x, train_y)
test = torch.utils.data.TensorDataset(test_x, test_y)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

# define model
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_cells = num_layers
        self.output_dim = output_dim
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_cells, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_cells, x.size(0), self.hidden_dim).to(device)

        # x, h_s = self.lstm(x, (h0, c0))
        x, h_s = self.gru(x, h0)
        x = self.fc1(x[:, -1, :])
        x = nn.functional.relu(x)
        x = self.fc2(x)

        return x

# define loss and optimizer
net = Net(test_x[0][0].shape[0], hidden_dim, output_dim, num_layers).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# define measurement
def MAE(y, y_hat):
    return np.mean(np.abs(y - y_hat))

train_losses = []
# train
net.train()
for epoch in range(n_epochs):
    train_batch = iter(train_loader)
    for inputs, labels in train_batch:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = torch.sqrt(loss_fn(outputs, labels.float()) + 1e-6)
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

torch.save(net.state_dict(), "cfo_rnn.pth")

# plot loss
if plot:
    fig_l = plt.figure(figsize=(8, 6))
    plt.plot(train_losses)
    plt.title("Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

# test
# net.load_state_dict(torch.load("cfo_rnn.pth"))
net.eval()
test_outputs = []
test_labels = []
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    test_outputs.append(scaler_y.inverse_transform(net(inputs).cpu().detach().numpy()))
    test_labels.append(scaler_y.inverse_transform(labels.cpu().detach().numpy()))

# measurements
test_outputs = np.array(test_outputs).squeeze().reshape(1, -1).squeeze()
test_labels = np.array(test_labels).squeeze().reshape(1, -1).squeeze()
mae = MAE(test_outputs, test_labels)
print(f"MAE: {mae.item()}")

# plot prediction results
if plot:
    fig_a = plt.figure(figsize=(8, 6))
    plt.plot(test_labels, 'b')
    plt.title("CFO estimation (label)")
    plt.xlabel("Samples")
    plt.ylabel("CFO")
    plt.show()

    fig_o = plt.figure(figsize=(8, 6))
    plt.plot(test_outputs, 'r')
    plt.title("CFO estimation (pred)")
    plt.xlabel("Samples")
    plt.ylabel("CFO")
    plt.show()
