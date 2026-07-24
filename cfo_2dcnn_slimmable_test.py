import os
import gc
import numpy as np
from scipy.signal import detrend

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--cutout", type=bool, default=True)
parser.add_argument("--auto", type=bool, default=True)
parser.add_argument("--num_lost", type=int, default=1)  # if auto is False, 1 ~ 5

# slimmable settings
parser.add_argument("--widths", type=str, default="0.25,0.5,0.75,1.0",
                    help="Comma-separated width multipliers used in the trained slimmable model.")
parser.add_argument("--test_width", type=float, default=1.0,
                    help="Single width to test. If --eval_all_widths is set, this is ignored.")
parser.add_argument("--eval_all_widths", type=bool, default=False,
                    help="If True, evaluate MAE at all widths in --widths.")

# gpu
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(gpu=True)
args = parser.parse_args()

seed = torch.initial_seed()
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

num_threads = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
torch.set_num_threads(num_threads)

fname_test = "/home/leehyunjong/Wi-Fi_Preambles/stfcfo/wireless/" \
             "WiFi_10MHz_Preambles_wireless_cfo_test_-3dB.txt"

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
    phase = np.angle(dcr).astype(np.float32)

    test_signals.append((phase, float(line_label)))

test_x = torch.tensor(np.stack([i[0] for i in test_signals]), device=device)
test_y = torch.tensor(np.expand_dims(np.stack([i[1] for i in test_signals]), 1), device=device)

# Obtain y_min and y_max from test labels (for denormalization)
y_max = test_y.max().item()
y_min = test_y.min().item()

# data loader
test = torch.utils.data.TensorDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)


def MAE(y, y_hat):
    return np.mean(np.abs(y - y_hat))


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

        self.fc1 = nn.Linear(self.max_c * 40, self.max_fc1)
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
        x = self.flatten(x)

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


MODEL_PATH = "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/incomplete/" \
             "cfo_2dcnn_wireless_slimmable.pth"

net = SlimmableNet2D(WIDTHS).to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.eval()


def run_eval(width: float):
    net.set_width(width)
    test_outputs = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
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

            test_outputs.append(net(inputs).cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_outputs = np.array(test_outputs).squeeze().reshape(1, -1).squeeze() * (y_max - y_min) + y_min
    test_labels = np.array(test_labels).squeeze().reshape(1, -1).squeeze()

    mae = MAE(test_outputs, test_labels)

    print(f"[Slimmable-2D] width={net.active_width:.2f} | Average MAE: {mae.item()}")
    return mae


if args.eval_all_widths:
    for w in WIDTHS:
        run_eval(w)
else:
    run_eval(args.test_width)
