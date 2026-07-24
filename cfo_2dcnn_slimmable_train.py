import os
import gc
import numpy as np
from scipy.signal import detrend

import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer as torch_optim

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--cutout", type=bool, default=True)
parser.add_argument("--auto", type=bool, default=True)
parser.add_argument("--num_lost", type=int, default=1)  # if auto is False, 1 ~ 5

# training
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--schedular_patience", type=int, default=2)
parser.add_argument("--gradient_max_norm", type=float, default=5.0)
parser.add_argument('--early_stop', type=bool, default=True)
parser.add_argument('--es_patience', type=int, default=10)

# slimmable settings
parser.add_argument("--widths", type=str, default="0.25,0.5,0.75,1.0",
                    help="Comma-separated width multipliers (prefix channels/neurons are used).")
parser.add_argument("--val_width", type=float, default=1.0,
                    help="Width multiplier used for validation / early-stopping (default: 1.0).")
parser.add_argument("--width_sampling", type=str, default="random",
                    choices=["random", "max", "min"],
                    help="How to sample width during training iterations.")

# gpu
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(gpu=True)
args = parser.parse_args()

seed = torch.initial_seed()
random.seed(seed)
np.random.seed(seed % (2**32 - 1))

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

num_threads = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
torch.set_num_threads(num_threads)

# load data and change i to j (complex number)
fname_train = "/home/leehyunjong/Wi-Fi_Preambles/stfcfo/wireless/" \
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
    line_data = line[0:160]
    line_label = np.real(line[-1])
    dcr = detrend(line_data - np.mean(line_data))
    phase = np.angle(dcr).astype(np.float32)

    train_signals.append((phase, float(line_label)))

# Apply min–max normalization to Y values
all_y_np = np.stack([i[1] for i in train_signals])
y_min = all_y_np.min()
y_max = all_y_np.max()
normalized_y = (all_y_np - y_min) / (y_max - y_min)

# Create TensorDataset for training/validation
all_x = torch.tensor(np.stack([i[0] for i in train_signals]), device=device)
all_y = torch.tensor(np.expand_dims(normalized_y, axis=1), device=device, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(all_x, all_y)
dataset_size = len(dataset)

# Split dataset into train (80%) and validation (20%)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def _parse_widths(widths_str: str):
    ws = [float(w.strip()) for w in widths_str.split(",") if len(w.strip()) > 0]
    ws = sorted(list(dict.fromkeys(ws)))
    if len(ws) == 0:
        raise ValueError("Empty widths list. Provide e.g., --widths 0.25,0.5,0.75,1.0")
    for w in ws:
        if w <= 0.0 or w > 1.0:
            raise ValueError(f"Width multipliers must be in (0,1], got {w}")
    return ws


WIDTHS = _parse_widths(args.widths)


class SlimmableNet2D(nn.Module):
    """
    Slimmable 2D CNN (single shared network executable at multiple widths).
    - Prefix slicing of channels/neurons based on a global width multiplier.
    - Switchable BatchNorm for the Conv layer.
    """
    def __init__(self, widths):
        super().__init__()
        self.widths = widths
        self.max_c = 16
        self.max_fc1 = 512
        self.max_fc2 = 256

        self.conv = nn.Conv2d(1, self.max_c, 3, padding=1, stride=1, bias=True)
        self.bn_list = nn.ModuleList([nn.BatchNorm2d(max(1, int(self.max_c * w))) for w in self.widths])

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(self.max_c * 40, self.max_fc1)  # 16*(5*8)=640
        self.fc2 = nn.Linear(self.max_fc1, self.max_fc2)
        self.fc3 = nn.Linear(self.max_fc2, 1)

        self.active_width = 1.0

    def set_width(self, width: float):
        width = float(width)
        nearest = min(self.widths, key=lambda w: abs(w - width))
        self.active_width = nearest

    def _width_idx(self):
        return self.widths.index(self.active_width)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, 10, 16)

        w = self.active_width
        c_out = max(1, int(self.max_c * w))
        fc1_out = max(1, int(self.max_fc1 * w))
        fc2_out = max(1, int(self.max_fc2 * w))

        w_conv = self.conv.weight[:c_out].contiguous()
        b_conv = self.conv.bias[:c_out].contiguous() if self.conv.bias is not None else None
        x = F.conv2d(
            x, w_conv, b_conv,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )

        bn = self.bn_list[self._width_idx()]
        x = bn(x)
        x = F.relu(x)

        x = self.pool(x)
        x = self.flatten(x)  # (B, c_out*40)

        in_dim = x.shape[1]

        w1 = self.fc1.weight[:fc1_out, :in_dim].contiguous()
        b1 = self.fc1.bias[:fc1_out].contiguous()
        x = F.linear(x, w1, b1)
        x = F.relu(x)

        w2 = self.fc2.weight[:fc2_out, :fc1_out].contiguous()
        b2 = self.fc2.bias[:fc2_out].contiguous()
        x = F.linear(x, w2, b2)
        x = F.relu(x)

        w3 = self.fc3.weight[:, :fc2_out].contiguous()
        b3 = self.fc3.bias
        x = F.linear(x, w3, b3)

        return x


def sample_train_width():
    if args.width_sampling == "max":
        return max(WIDTHS)
    if args.width_sampling == "min":
        return min(WIDTHS)
    return random.choice(WIDTHS)


# define loss and optimizer
net = SlimmableNet2D(WIDTHS).to(device)
net.set_width(1.0)

loss_fn = nn.MSELoss()
optimizer = torch_optim.Lookahead(optim.RAdam(net.parameters(), lr=learning_rate))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=args.schedular_patience
)

SAVE_PATH = "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/incomplete/" \
            "cfo_2dcnn_wireless_slimmable.pth"

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

for epoch in range(n_epochs):
    net.train()
    train_losses = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if args.cutout:
            input_mask = torch.ones_like(inputs, device=device)
            for i in range(inputs.size(0)):
                if args.auto:
                    pos = torch.randint(1, 6, (1,), device=device).item()  # 1~5
                else:
                    pos = args.num_lost
                input_mask[i, :32 * (pos - 1)] = 0
            inputs = inputs * input_mask

        net.set_width(sample_train_width())

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = torch.sqrt(loss_fn(outputs, labels.float()) + 1e-8)
        loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.gradient_max_norm)
        optimizer.step()

        train_losses.append(loss.item())

    avg_train_loss = sum(train_losses) / len(train_losses) if len(train_losses) > 0 else 0.0

    # Validation (fixed width)
    net.eval()
    net.set_width(args.val_width)

    val_losses = []
    val_mae = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if args.cutout:
                input_mask = torch.ones_like(inputs, device=device)
                for i in range(inputs.size(0)):
                    if args.auto:
                        pos = torch.randint(1, 6, (1,), device=device).item()  # 1~5
                    else:
                        pos = args.num_lost
                    input_mask[i, :32 * (pos - 1)] = 0
                inputs = inputs * input_mask

            outputs = net(inputs)
            loss_val = torch.sqrt(loss_fn(outputs, labels.float()) + 1e-8)
            mae_val = torch.mean(torch.abs(outputs - labels))
            val_losses.append(loss_val.item())
            val_mae.append(mae_val.item())

    avg_val_loss = sum(val_losses) / len(val_losses) if len(val_losses) > 0 else float("inf")
    avg_val_mae = sum(val_mae) / len(val_mae) if len(val_mae) > 0 else float("inf")
    denorm_val_mae = avg_val_mae * (y_max - y_min)

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}/{n_epochs} - Train RMSE: {avg_train_loss:.4f}, "
          f"Val RMSE@w={net.active_width:.2f}: {avg_val_loss:.4f}, Val MAE (denorm): {denorm_val_mae:.4f}")

    if args.early_stop:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.es_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if best_model_state is not None:
    net.load_state_dict(best_model_state, strict=True)

torch.save(net.state_dict(), SAVE_PATH)
print(f"Saved slimmable 2D CNN model to: {SAVE_PATH}")
