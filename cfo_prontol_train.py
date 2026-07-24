import os
import gc
import numpy as np
from scipy.signal import detrend

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as torch_optim

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--cutout", type=bool, default=True)
parser.add_argument("--auto", type=bool, default=True)
parser.add_argument("--num_lost", type=int, default=1) # 1 ~ 5
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--schedular_patience", type=int, default=2)
parser.add_argument("--gradient_max_norm", type=float, default=5.0)
parser.add_argument('--early_stop', type=bool, default=True)
parser.add_argument('--es_patience', type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(gpu=True)
args = parser.parse_args()

seed = torch.initial_seed()
batch_size = args.batch_size
n_epochs = args.n_epochs
learning_rate = args.learning_rate
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
fname_train = "/home/leehyunjong/Wi-Fi_Preambles/ltfcfo/wireless/"\
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
    # Extract the last 'input_size' samples from the full preamble
    line_data = line[0:160]
    line_label = np.real(line[-1])
    dcr = detrend(line_data - np.mean(line_data))
    real = np.real(dcr).astype(np.float32)
    imag = np.imag(dcr).astype(np.float32)
    real_rms = np.sqrt(np.sum(np.power(np.abs(real), 2)) / 160)
    imag_rms = np.sqrt(np.sum(np.power(np.abs(imag), 2)) / 160)

    # Concatenate into flat 320-dim vector (first 160: real, next 160: imag)
    whole = np.stack([real / real_rms, imag / imag_rms], axis=0)
    train_signals.append((whole, float(line_label)))

# Apply min–max normalization to Y values
all_y_np = np.stack([i[1] for i in train_signals])
y_min = all_y_np.min()
y_max = all_y_np.max()
normalized_y = (all_y_np - y_min) / (y_max - y_min)

# Create TensorDataset for training/validation
all_x = torch.tensor(np.stack([i[0] for i in train_signals], axis=0), device=device)
all_y = torch.tensor(np.expand_dims(normalized_y, axis=1), device=device, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(all_x, all_y)
dataset_size = len(dataset)

# Split dataset into train (80%) and validation (20%)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# define model
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

# define loss and optimizer
net = Net().to(device)
loss_fn = nn.MSELoss()
optimizer = torch_optim.Lookahead(optim.RAdam(net.parameters(), lr=learning_rate))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                 patience=args.schedular_patience)

# train
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

# Training loop with validation (80:20 split)
for epoch in range(n_epochs):
    net.train()
    train_losses = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if args.cutout:
            # inputs: [B, 2, 160]
            input_mask = torch.ones_like(inputs, device=device)
            for i in range(inputs.size(0)):
                if args.auto:
                    pos = torch.randint(1, 6, (1,), device=device).item()  # 1~5
                else:
                    pos = args.num_lost

                input_mask[i, :, :32 * (pos - 1)] = 0

            inputs = inputs * input_mask

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = torch.sqrt(loss_fn(outputs, labels.float()) + 1e-8)
        loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.gradient_max_norm)
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = sum(train_losses) / len(train_losses)

    # Validation phase
    net.eval()
    val_losses = []
    val_mae = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if args.cutout:
                # inputs: [B, 2, 160]
                input_mask = torch.ones_like(inputs, device=device)
                for i in range(inputs.size(0)):
                    if args.auto:
                        pos = torch.randint(1, 6, (1,), device=device).item()  # 1~5
                    else:
                        pos = args.num_lost

                    input_mask[i, :, :32 * (pos - 1)] = 0

                inputs = inputs * input_mask

            outputs = net(inputs)
            loss_val = torch.sqrt(loss_fn(outputs, labels.float()) + 1e-8)
            mae_val = torch.mean(torch.abs(outputs - labels))
            val_losses.append(loss_val.item())
            val_mae.append(mae_val.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_val_mae = sum(val_mae) / len(val_mae)
    denorm_val_mae = avg_val_mae * (y_max - y_min)
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}/{n_epochs} - Val RMSE: {avg_val_loss:.4f}, Val MAE (denorm): {denorm_val_mae:.4f}")

    # Early stopping
    if args.early_stop:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = net.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.es_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                net.load_state_dict(best_model_state)
                torch.save(net.state_dict(),
                           "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/incomplete/cfo_prontol_wireless.pth")

                break

# torch.save(net.state_dict(), "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/incomplete/cfo_prontol_wireless.pth")
