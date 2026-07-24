import os
import sys
import gc
import numpy as np
from scipy.signal import detrend
import argparse

from spikingjelly.activation_based import functional, neuron, monitor
import torch
from collections import Counter

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--cutout", type=bool, default=True)
parser.add_argument("--auto", type=bool, default=True)
parser.add_argument("--num_lost", type=int, default=1)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(gpu=True)
args = parser.parse_args()

# Setup seeds and device
gc.collect()
torch.cuda.empty_cache()
if args.spare_gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.spare_gpu)
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
seed = torch.initial_seed()
if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
torch.set_num_threads(os.cpu_count() - 1)

# Import Net
sys.path.append("/SNN/CFO/train")
import cfo_scnn_train
sys.modules["__main__"] = cfo_scnn_train

# Load dataset
data_path = "/home/leehyunjong/Wi-Fi_Preambles/stfcfo/wireless/WiFi_10MHz_Preambles_wireless_cfo_test_rician_18dB.txt"
raw = np.loadtxt(data_path, dtype=str, delimiter='\t')
np.random.shuffle(raw)
for i in range(raw.shape[0]):
    raw[i] = [x.replace('i','j') for x in raw[i]]
raw = raw.astype(np.complex64)

# Preprocess
test_x, test_y = [], []
for line in raw:
    data, label = line[:160], np.real(line[-1])
    d = detrend(data - data.mean())
    real = np.real(d).astype(np.float32)
    imag = np.imag(d).astype(np.float32)
    real_rms = np.sqrt((real ** 2).mean())
    imag_rms = np.sqrt((imag ** 2).mean())
    test_x.append(np.stack([real/real_rms, imag/imag_rms], axis=0))
    test_y.append(float(label))

# Tensor and loader
test_x = torch.tensor(np.stack(test_x), device=device)
test_y = torch.tensor(test_y, device=device).unsqueeze(1)
from torch.utils.data import DataLoader, TensorDataset
loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

y_max, y_min = test_y.max().item(), test_y.min().item()

# Load model
model_file = "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/incomplete/cfo_scnn_wireless.pt"
net = torch.load(model_file, map_location=device, weights_only=False)
net.to(device)
net.eval()

# Prepare stats
num_conv = len(net.conv_blocks)
num_lin = len(net.linear_blocks)
stats = {
    'depth_conv': [],
    'depth_linear': [],
    'width_logits': [[] for _ in range(num_conv-1)],
    'expert_logits': [[] for _ in range(num_lin)]
}

# Hooks for depth gates
def hook_depth_conv(mod, inp, out):
    depth, *_ = out
    stats['depth_conv'].extend(depth.cpu().tolist())

def hook_depth_linear(mod, inp, out):
    depth, *_ = out
    stats['depth_linear'].extend(depth.cpu().tolist())

net.depth_gate_conv.register_forward_hook(hook_depth_conv)
net.depth_gate.register_forward_hook(hook_depth_linear)

# Optional: OutputMonitor if needed
fr_monitor = monitor.OutputMonitor(net, neuron.IFNode, lambda s: s.flatten(1).mean(1))
fr_monitor.clear_recorded_data()

# Run test
preds = []
labels = []
for x_batch, y_batch in loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    B = x_batch.size(0)

    # Conv-stage depth adaptation: manual replication of forward's conv stage and depth gating
    with torch.no_grad():
        x_conv = x_batch.view(B, 2, 10, 16)  # reshape as in Net.forward fileciteturn5file7
        feat = x_conv
        for blk in net.conv_blocks:
            feat = blk(feat)
        gap_feat = net.conv_gap(feat).view(B, -1)
        depth_conv_batch, _ = net.depth_gate_conv(gap_feat)
        stats['depth_conv'].extend(depth_conv_batch.cpu().tolist())

    # Apply cutout if enabled
    if args.cutout:
        mask = torch.ones_like(x_batch)
        for i in range(B):
            pos = torch.randint(1,6,(1,),device=device).item() if args.auto else args.num_lost
            mask[i,:, :32*(pos-1)] = 0
        x_batch = x_batch * mask

    # Forward pass captures linear depth (via hook), width & expert logits
    out = net(x_batch)
    y_hat = out[0]
    w_logits = out[-2]
    e_logits = out[-1]

    preds.append(y_hat.cpu().detach().numpy())
    labels.append(y_batch.cpu().detach().numpy())
    for i, w in enumerate(w_logits): stats['width_logits'][i].append(w.cpu().detach().numpy())
    for i, e in enumerate(e_logits): stats['expert_logits'][i].append(e.cpu().detach().numpy())

    functional.reset_net(net)

# Denormalize preds and flatten
all_preds = np.concatenate(preds,0).squeeze() * (y_max-y_min) + y_min
all_labels = np.concatenate(labels,0).squeeze()

# Print depth stats
print("\n-- Conv-stage Depth Gate Distribution --")
for d,c in Counter(stats['depth_conv']).items(): print(f"Depth {d/2}: {c/2}")
print("\n-- Linear-stage Depth Gate Distribution --")
for d,c in Counter(stats['depth_linear']).items(): print(f"Depth {d}: {c}")

# Width gate stats
print("\n-- Conv Width Filter Counts --")
for idx, wl in enumerate(stats['width_logits'], start=1):
    arr = np.vstack(wl)
    sel = np.argmax(arr,1)
    blk = net.conv_blocks[idx]
    widths, max_out = blk.core[0].widths, blk.core[0].max_out
    cnts = Counter([int(max_out*widths[i]) for i in sel])
    print(f"Block {idx}:")
    for num, cnt in cnts.items(): print(f" {num} filters: {cnt}")

# Expert stats
print("\n-- MoE Expert Selection --")
for idx, el in enumerate(stats['expert_logits']):
    arr = np.vstack(el)
    sel = np.argmax(arr,1)
    num_dnn = len(net.linear_blocks[idx].dnn_experts)
    print(f"Block {idx}: DNN {(sel<num_dnn).sum()}, SNN {(sel>=num_dnn).sum()}")

# Overall MAE
mae = np.mean(np.abs(all_preds-all_labels))
print(f"\n-- Overall MAE: {mae:.4f} --")
