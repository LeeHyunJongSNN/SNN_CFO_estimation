import os
import gc
import numpy as np
from scipy.signal import detrend

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as torch_optim
import torch.nn.utils.prune as prune

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--cutout", type=bool, default=True)
parser.add_argument("--auto", type=bool, default=True)
parser.add_argument("--num_lost", type=int, default=1)  # if auto is False, 1 ~ 5

parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--schedular_patience", type=int, default=2)
parser.add_argument("--gradient_max_norm", type=float, default=5.0)
parser.add_argument('--early_stop', type=bool, default=True)
parser.add_argument('--es_patience', type=int, default=10)

# pruning settings (magnitude-based, global unstructured)
parser.add_argument("--prune_ratio", type=float, default=0.8,
                    help="Global unstructured pruning ratio (0~1). Example: 0.8 prunes 80% weights.")
parser.add_argument("--prune_epoch", type=int, default=50,
                    help="Epoch to apply pruning mask (start fine-tuning after this epoch).")
parser.add_argument("--start_early_stop_after_prune", type=bool, default=True,
                    help="If True, early-stopping starts after prune_epoch so pruning definitely happens.")

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv1d(1, 16, 3, padding=1, stride=1)
        self.batch_norm = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, -1)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x


def apply_global_magnitude_pruning(model: nn.Module, amount: float):
    """
    Global unstructured magnitude pruning (L1Unstructured) applied to all Conv/Linear weights.
    This follows the classic magnitude pruning pipeline used in early network pruning literature.
    """
    parameters_to_prune = []
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            parameters_to_prune.append((m, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    return parameters_to_prune


def remove_pruning_reparam(parameters_to_prune):
    for (m, name) in parameters_to_prune:
        try:
            prune.remove(m, name)
        except ValueError:
            # already removed
            pass


def compute_global_sparsity(model: nn.Module) -> float:
    total = 0
    zero = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            w = m.weight.detach()
            total += w.numel()
            zero += (w == 0).sum().item()
    return zero / total if total > 0 else 0.0


net = Net().to(device)
loss_fn = nn.MSELoss()
optimizer = torch_optim.Lookahead(optim.RAdam(net.parameters(), lr=learning_rate))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=args.schedular_patience
)

SAVE_PATH = "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/incomplete/" \
            "cfo_1dcnn_wireless_pruned.pth"

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None
pruned = False
parameters_to_prune = []

for epoch in range(n_epochs):
    net.train()
    train_losses = []

    # Apply pruning once at prune_epoch
    if (not pruned) and (epoch == args.prune_epoch):
        parameters_to_prune = apply_global_magnitude_pruning(net, amount=args.prune_ratio)
        pruned = True
        print(f"[Pruning] Applied global magnitude pruning at epoch {epoch} (ratio={args.prune_ratio}). "
              f"Current sparsity={compute_global_sparsity(net)*100:.2f}%")

        # (optional) reset scheduler/early-stop counters after pruning
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

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

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = torch.sqrt(loss_fn(outputs, labels.float()) + 1e-8)
        loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.gradient_max_norm)
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = sum(train_losses) / len(train_losses) if len(train_losses) > 0 else 0.0

    # Validation
    net.eval()
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

    sparsity = compute_global_sparsity(net) * 100.0
    print(f"Epoch {epoch+1}/{n_epochs} - Train RMSE: {avg_train_loss:.4f}, "
          f"Val RMSE: {avg_val_loss:.4f}, Val MAE (denorm): {denorm_val_mae:.4f}, "
          f"Sparsity: {sparsity:.2f}%")

    # Early stopping (optionally starts after pruning)
    allow_es = True
    if args.start_early_stop_after_prune:
        allow_es = pruned  # only after pruning

    if args.early_stop and allow_es:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.es_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

# Load best state (if tracked)
if best_model_state is not None:
    net.load_state_dict(best_model_state, strict=True)

# Remove pruning re-parameterization so state_dict is loadable by the original Net class
if pruned and len(parameters_to_prune) > 0:
    remove_pruning_reparam(parameters_to_prune)
    print(f"[Pruning] Removed pruning re-parameterization. Final sparsity={compute_global_sparsity(net)*100:.2f}%")

torch.save(net.state_dict(), SAVE_PATH)
print(f"Saved pruned 1D CNN model to: {SAVE_PATH}")
