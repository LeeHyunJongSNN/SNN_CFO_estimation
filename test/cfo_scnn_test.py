import os
import sys
import gc
import numpy as np
from scipy.signal import detrend
import argparse

from spikingjelly.activation_based import functional, neuron, monitor
import torch

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--cutout", type=bool, default=False)
parser.add_argument("--auto", type=bool, default=True)
parser.add_argument("--num_lost", type=int, default=1) # if auto is False, 1 ~ 5
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(gpu=True)
args = parser.parse_args()

seed = torch.initial_seed()
batch_size = args.batch_size
gpu = args.gpu
spare_gpu = args.spare_gpu

# Set up GPU usage
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

# Add the directory containing the training file to sys.path
# (Assuming cfo_scnn_train.py is located in "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO")
sys.path.append("/SNN/CFO/train")
import cfo_scnn_train  # This module contains the Net class

# Trick pickle to find the Net class by mapping __main__ to the training module
sys.modules["__main__"] = cfo_scnn_train

# Load test dataset from file
fname_test = ("/home/leehyunjong/Wi-Fi_Preambles/stfcfo/wireless/"
              "WiFi_10MHz_Preambles_wireless_cfo_test_-3dB.txt")

raw_test = np.loadtxt(fname_test, dtype='str', delimiter='\t')
np.random.shuffle(raw_test)
for i in range(len(raw_test)):
    for j in range(len(raw_test[i])):
        raw_test[i][j] = raw_test[i][j].replace('i', 'j')
raw_test = raw_test.astype(np.complex64)

# Remove DC offsets and prepare test signals
test_signals = []
for line in raw_test:

    # Extract the last 'input_size' samples from the full preamble
    line_data = line[0:160]
    line_label = np.real(line[-1])
    dcr = detrend(line_data - np.mean(line_data))
    real = np.real(dcr).astype(np.float32)
    imag = np.imag(dcr).astype(np.float32)
    real_rms = np.sqrt(np.sum(np.power(np.abs(real), 2)) / 160)
    imag_rms = np.sqrt(np.sum(np.power(np.abs(imag), 2)) / 160)

    # Concatenate into flat 320-dim vector (first 160: real, next 160: imag)
    # whole = np.stack([real, imag], axis=0)
    whole = np.stack([real / real_rms, imag / imag_rms], axis=0)
    test_signals.append((whole, float(line_label)))

test_x = torch.tensor(np.stack([i[0] for i in test_signals]), device=device)
test_y = torch.tensor(np.expand_dims(np.stack([i[1] for i in test_signals]), axis=1), device=device)

# Obtain y_min and y_max from test labels (for denormalization)
y_max = test_y.max().item()
y_min = test_y.min().item()

# Create test DataLoader
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Define measurement function
def MAE(y, y_hat):
    return np.mean(np.abs(y - y_hat))

# Load the entire saved model (the entire model was saved as .pt)
model_path = "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/complete/cfo_scnn_wireless.pt"
net = torch.load(model_path, map_location=device, weights_only=False)
net.to(device)

# Define firing rate monitor
def cal_firing_rate(s_seq: torch.Tensor):
    return s_seq.flatten(1).mean(1)
fr_monitor = monitor.OutputMonitor(net, neuron.IFNode, cal_firing_rate)
fr_monitor.clear_recorded_data()

net.eval()
test_outputs = []
test_labels = []

# Testing loop
for inputs, labels in test_loader:
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

    # test_outputs.append(net(inputs).cpu().detach().numpy())
    test_outputs.append(net(inputs)[0].cpu().detach().numpy())
    test_labels.append(labels.cpu().detach().numpy())
    functional.reset_net(net)

# Denormalize predictions: MAE denormalization for differences only requires multiplication by (y_max - y_min)
test_outputs = np.array(test_outputs).squeeze().reshape(1, -1).squeeze() * (y_max - y_min) + y_min
test_labels = np.array(test_labels).squeeze().reshape(1, -1).squeeze()

# Compute MAE for segments and overall
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

print(f"MAE segment 1: {mae_1.item()}")
print(f"MAE segment 2: {mae_2.item()}")
print(f"MAE segment 3: {mae_3.item()}")
print(f"MAE segment 4: {mae_4.item()}")
print(f"MAE segment 5: {mae_5.item()}")
print(f"MAE segment 6: {mae_6.item()}")
print(f"MAE segment 7: {mae_7.item()}")
print(f"MAE segment 8: {mae_8.item()}")
print(f"MAE segment 9: {mae_9.item()}")
print(f"MAE segment 10: {mae_10.item()}")
print(f"Average MAE: {mae.item()}")

# Compute average spiking rate from the monitor records
# spike_rate = fr_monitor.records
# layer_avg = [torch.mean(rate).item() for rate in spike_rate]
#
# if "ConvFC1FC2" in network_type:
#     l_size = 3
# elif "ConvFC1" in network_type:
#     l_size = 2
# elif "ConvFC2" in network_type:
#     l_size = 2
# elif "FC1FC2" in network_type:
#     l_size = 2
# else:
#     l_size = 1
#
# layer_avg = np.mean(np.array(layer_avg).reshape(-1, l_size), 0)
# print(f"Average spiking rate: {layer_avg}")
