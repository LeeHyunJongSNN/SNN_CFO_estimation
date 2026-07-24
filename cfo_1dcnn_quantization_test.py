import os
import gc
import numpy as np
from scipy.signal import detrend

import argparse

import torch
import torch.nn as nn
import torch.ao.quantization as tq

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--cutout", type=bool, default=True)
parser.add_argument("--auto", type=bool, default=True)
parser.add_argument("--num_lost", type=int, default=1)  # if auto is False, 1 ~ 5

# quantization settings
parser.add_argument("--quant_engine", type=str, default="fbgemm",
                    help="Quantization backend: typically 'fbgemm' (x86) or 'qnnpack' (ARM).")

# NOTE: quantized inference runs on CPU (int8 operators are CPU backends).
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(gpu=False)
args = parser.parse_args()

seed = torch.initial_seed()
batch_size = args.batch_size

# set up (force CPU for quantized model)
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cpu")

num_threads = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
torch.set_num_threads(num_threads)

# quantization backend
supported_engines = torch.backends.quantized.supported_engines
engine = args.quant_engine if args.quant_engine in supported_engines else supported_engines[0]
torch.backends.quantized.engine = engine
print(f"[Quantization] Using quantization engine: {engine} (supported={supported_engines})")

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


class QuantNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = tq.QuantStub()

        self.conv = nn.Conv1d(1, 16, 3, padding=1, stride=1)
        self.batch_norm = nn.BatchNorm1d(16)
        self.relu0 = nn.ReLU()

        self.pool = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1280, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 1)

        self.dequant = tq.DeQuantStub()

    def fuse_model(self):
        tq.fuse_modules(self, [['conv', 'batch_norm', 'relu0'],
                               ['fc1', 'relu1'],
                               ['fc2', 'relu2']], inplace=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, -1)

        x = self.quant(x)

        x = self.conv(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.dequant(x)
        return x


MODEL_PATH = "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/incomplete/" \
             "cfo_1dcnn_wireless_qat_int8.pth"

# Build a quantized model instance (same conversion steps as training)
float_model = QuantNet1D()
float_model.eval()
float_model.fuse_model()
float_model.train()
float_model.qconfig = tq.get_default_qat_qconfig(engine)
tq.prepare_qat(float_model, inplace=True)

# Run a tiny dummy batch to initialize observers (avoids qparams warnings during convert)
with torch.no_grad():
    float_model(torch.zeros(1, 160))

quantized_model = tq.convert(float_model.eval(), inplace=False)
quantized_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'), strict=True)
quantized_model.eval()

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

        test_outputs.append(quantized_model(inputs).cpu().numpy())
        test_labels.append(labels.cpu().numpy())

test_outputs = np.array(test_outputs).squeeze().reshape(1, -1).squeeze() * (y_max - y_min) + y_min
test_labels = np.array(test_labels).squeeze().reshape(1, -1).squeeze()

mae = MAE(test_outputs, test_labels)
print(f"[QAT-int8-1D] Average MAE: {mae.item()}")
