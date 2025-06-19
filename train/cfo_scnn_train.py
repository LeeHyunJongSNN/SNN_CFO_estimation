import os
import gc
import re
import numpy as np
from scipy.signal import detrend
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_optimizer as torch_optim
from spikingjelly.activation_based import neuron, functional, surrogate

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--cutout", type=bool, default=False)
parser.add_argument("--auto", type=bool, default=True)
parser.add_argument("--num_lost", type=int, default=1) # if auto is False, 1 ~ 5
parser.add_argument("--conv_channels", type=int, default=[64, 64])
parser.add_argument("--linear_dims", type=int, default=[64, 32, 32])
parser.add_argument("--num_blocks_1", type=int, default=2)
parser.add_argument("--num_blocks_2", type=int, default=2)
parser.add_argument("--alpha", type=float, default=1.0)  # 1.0
parser.add_argument("--beta", type=float, default=0.5)   # 0.5
parser.add_argument("--gamma", type=float, default=0.5)  # 0.5
parser.add_argument("--delta", type=float, default=0.1)  # 0.1
parser.add_argument("--eta", type=float, default=2.0)
parser.add_argument("--epsilon", type=float, default=0.1)
parser.add_argument("--temp", type=float, default=3.0)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--schedular_patience", type=int, default=2)
parser.add_argument("--gradient_max_norm", type=float, default=5.0)
parser.add_argument('--early_stop', type=bool, default=True)
parser.add_argument('--es_patience', type=int, default=10)
parser.add_argument("--num_steps", type=int, default=2)
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

# Load training/validation data from file
fname_train = "/home/leehyunjong/Wi-Fi_Preambles/stfcfo/wireless/WiFi_10MHz_Preambles_wireless_cfo_train.txt"
raw_train = np.loadtxt(fname_train, dtype='str', delimiter='\t')
np.random.shuffle(raw_train)
for i in range(len(raw_train)):
    for j in range(len(raw_train[i])):
        raw_train[i][j] = raw_train[i][j].replace('i', 'j')
raw_train = raw_train.astype(np.complex64)

# Remove DC offsets and prepare training signals
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
    # whole = np.stack([real, imag], axis=0)
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

def _hard_gate_indices(logits, tau):
    """Straight-through Gumbel-Softmax → one-hot → indices"""
    gate = F.gumbel_softmax(logits, tau=tau, hard=True)       # [B, N]
    idx  = gate.argmax(dim=1)                                 # [B]
    return gate, idx

def _split_expert_counts(total_blocks: int):
    """
    return: (num_dnn, num_snn)
      even  -> (K/2, K/2)
      odd   -> (ceil(K/2), floor(K/2))  # DNN 하나 더
    """
    num_dnn = (total_blocks + 1) // 2      # 올림
    num_snn = total_blocks // 2            # 내림
    return num_dnn, num_snn

@torch.no_grad()
def anneal_and_clamp_tau(model, sched_tau, tau_max=None):
    """
    Sched_tau: 이번 epoch 목표 하한값
    tau_max  : 선택. 상한을 주고 싶으면 숫자 넣기
    """
    for m in model.modules():
        if hasattr(m, 'tau'):
            # optimizer step으로 갱신된 값을 가져온 뒤 ↓ 하한/상한 클램프
            if tau_max is None:
                m.tau.data.clamp_(min=sched_tau)
            else:
                m.tau.data.clamp_(min=sched_tau, max=tau_max)

# ───────── Energy constants (pJ) ─────────
E_MAC = 3.1        # Multiply
E_AC  = 0.1        # Accumulate

# SNN sparsity 평균값
T_STEPS        = 2        # simulation timesteps
SPIKE_RATE_AVG = 0.18     # 평균 spike rate

# ───────── pJ-리스트용 에너지 계산 함수 ─────────
def compute_expected_energy_precalc(gate_logits, energy_costs_pJ, tau=1.0):
    """
    energy_costs_pJ : 이미 pJ 단위로 계산된 [K] 리스트
    gate_logits     : (B, K)
    반환값           : 배치 평균 에너지 (J 단위)
    """
    import torch.nn.functional as F  # 로컬 import
    probs = F.gumbel_softmax(gate_logits, tau=tau, hard=False)  # (B,K)
    e_pJ  = gate_logits.new_tensor(energy_costs_pJ)             # (K)

    return (probs * e_pJ).sum(dim=1).mean() * 1e-12             # → J

class DSConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DSConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class DSConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, bias: bool = False):
        super().__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, channels, L)
        b, c, _ = x.size()

        # Global average pooling: (batch, channels)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)

        return x * y

class DSConv1dSE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, reduction=4):
        super(DSConv1dSE, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.se = SEBlock1D(out_channels, reduction)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)

        return out

class SEBlock2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)   # squeeze (B, C)
        y = self.fc(y).view(b, c, 1, 1)   # excite  (B, C,1,1)
        return x * y                      # scale

class DSConv2dSE(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int, kernel_size, stride=1, padding=0,
                 bias: bool = False, reduction: int = 4):
        super().__init__()

        # depthwise: groups=in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=bias)

        # pointwise: 1×1 conv
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.se   = SEBlock2d(out_channels, reduction)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)  # (B, Cout, H, W)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)

        return out

class SlimmableDSConv2dSE(nn.Module):
    """
    DSConv2d + SE + 폭(slimmable) + Learned 1×1 up-projection.
    Args:
        in_channels : 입력 채널(Cin)
        max_out     : point-wise 최대 출력 채널(Cout_max)
        kernel_size : depth-wise 커널 크기 (3 추천)
        widths      : 폭 비율 리스트, ex (0.25, 0.5, 0.75, 1.0)
    """
    def __init__(self, in_channels, max_out, kernel_size=3,
                 widths=(0.25, 0.5, 0.75, 1.0),
                 stride=1, padding=1, bias=False, reduction=4):
        super().__init__()
        self.widths   = widths
        self.max_out  = max_out

        # (1) depth-wise
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=bias)

        # (2) point-wise full weight (max_out filters)
        self.pointwise = nn.Conv2d(in_channels, max_out, kernel_size=1, bias=bias)

        # (3) 폭별 BN & SE
        self.bn_list = nn.ModuleList([nn.BatchNorm2d(int(max_out * r)) for r in widths])
        self.relu    = nn.ReLU(inplace=False)
        self.se_list = nn.ModuleList([SEBlock2d(int(max_out * r), reduction) for r in widths])

        # (4) 폭 게이트
        self.gate_fc = nn.Linear(in_channels, len(widths))
        self.tau     = nn.Parameter(torch.tensor(1.0))       # learnable temperature

        # (5) Learned 1×1 up-projection (max_out → max_out)
        self.up = nn.Conv2d(max_out, max_out, kernel_size=1, bias=False)
        with torch.no_grad():
            eye = torch.eye(max_out)                          # identity init
            self.up.weight.copy_(eye.view(max_out, max_out, 1, 1))

    # ------------------------------------------------------
    def forward(self, x):
        # depth-wise
        out = self.depthwise(x)                               # (B,Cin,H,W)

        # 폭 선택
        logits = self.gate_fc(F.adaptive_avg_pool2d(out,1).flatten(1))
        gate   = F.gumbel_softmax(logits, tau=self.tau, hard=True)
        idx    = gate.float().mean(0).argmax().item()         # 배치 평균으로 1개 선택
        keep   = int(self.max_out * self.widths[idx])         # 활성 채널 수

        # point-wise slice
        w_pw = self.pointwise.weight[:keep]                   # (keep,Cin,1,1)
        out  = F.conv2d(out, w_pw, None, 1, 0)                # (B,keep,H,W)

        # 전용 BN + SE
        out  = self.bn_list[idx](out)
        out  = self.relu(out)
        out  = self.se_list[idx](out)

        # zero-pad → (B,max_out,H,W)
        if keep < self.max_out:
            pad = out.new_zeros(out.size(0), self.max_out - keep, *out.shape[2:])
            out = torch.cat([out, pad], dim=1)

        # learned 1×1 up-projection  (초깃값은 identity, 훈련 중 fine-tune)
        out = self.up(out)

        self.last_width_logits = logits

        return out

class DepthGate(nn.Module):
    def __init__(self, in_feat, max_depth=3, init_tau=1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(init_tau))
        self.fc  = nn.Linear(in_feat, max_depth)  # depth=1,2,3 중 택1

    def forward(self, feat):        # feat: ConvMoE flatten or pooled
        logits = self.fc(feat)      # (B, D)
        probs   = F.gumbel_softmax(logits, tau=self.tau.clamp(min=0.5), hard=True)
        depth  = probs.argmax(dim=1) + 1  # 1~D

        return depth, probs, logits

class LinearBlockWithDynamicGate(nn.Module):
    """
        Pre-gating hard-gate MoE for linear experts (half DNN, half SNN).
        """

    def __init__(self, in_features, out_features,
                 num_blocks_linear, num_steps, init_tau=1.0):
        super().__init__()
        num_dnn, num_snn = _split_expert_counts(num_blocks_linear)
        self.num_steps = num_steps
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))

        # experts
        self.dnn_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, out_features),
                          nn.ReLU())
            for _ in range(num_dnn)
        ])
        self.snn_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, out_features),
                          neuron.IFNode(v_threshold=1., v_reset=0.,
                                        surrogate_function=surrogate.ATan()))
            for _ in range(num_snn)
        ])

        self.gate_linear = nn.Linear(in_features, num_blocks_linear)

    # -------------------------------------------
    def forward(self, x):  # x: [B, Fin]
        b = x.size(0)
        logits = self.gate_linear(x)  # [B, N]
        _, idx = _hard_gate_indices(logits, self.tau)

        out_k_list, sel_list = [], []

        # DNN
        for k, expert in enumerate(self.dnn_experts):
            sel = (idx == k).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            out_k = expert(x[sel])
            out_k_list.append(out_k)
            sel_list.append(sel)

        # SNN
        offset = len(self.dnn_experts)
        for k, expert in enumerate(self.snn_experts):
            sel = (idx == offset + k).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            spk_sum = 0
            for _ in range(self.num_steps):
                spk_sum += expert(x[sel])
                functional.reset_net(expert)
            out_k = spk_sum / self.num_steps
            out_k_list.append(out_k)
            sel_list.append(sel)

        # re-assemble
        feat_dim = out_k_list[0].size(1)
        out = x.new_zeros(b, feat_dim)  # (batch, 64)

        for sel, out_k in zip(sel_list, out_k_list):
            out[sel] = out_k

        self.last_expert_logits = logits

        return out

class ConvMicroBlock(nn.Module):
    """DSConv2dSE → BN → ReLU → MaxPool2d(2)"""
    def __init__(self, conv_in, conv_out):
        super().__init__()
        self.core = nn.Sequential(
            DSConv2dSE(conv_in, conv_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)          # 10×16 → 5×8 → 2×4 ...
        )

    def forward(self, x):
        return self.core(x)

class SlimmableConvMicroBlock(nn.Module):
    """SlimmableDSConv2dSE → BN → ReLU → MaxPool2d(2)"""
    def __init__(self, conv_in, conv_out):
        super().__init__()
        self.core = nn.Sequential(
            SlimmableDSConv2dSE(conv_in, conv_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(conv_out),
            # nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)          # 10×16 → 5×8 → 2×4 ...
        )

    def forward(self, x):
        return self.core(x)

class DepthGateConv(nn.Module):
    def __init__(self, in_channels, max_depth=3, init_tau=1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(init_tau))
        self.fc  = nn.Linear(in_channels, max_depth)   # hard=True 사용

    def forward(self, feat):           # feat: (B, Cin)
        logits = self.fc(feat)
        gate   = F.gumbel_softmax(logits, tau=self.tau, hard=True)
        depth  = gate.argmax(dim=1) + 1          # 1~D

        return depth, gate

class Net(nn.Module):
    """
    conv_channels : e.g. [64, 64]           (length = #conv blocks)
    linear_dims   : e.g. [64, 32, 32]       (len = #linear blocks + 1)
                   ※ linear_dims[0] must equal conv_channels[-1]
    """
    def __init__(self,
                 conv_channels=args.conv_channels,
                 linear_dims=args.linear_dims,
                 number_of_blocks_1=args.num_blocks_1,
                 number_of_blocks_2=args.num_blocks_2):
        super().__init__()

        # ---------- Sanity check ----------
        assert len(conv_channels) >= 1,           "`conv_channels` must have ≥1 element"
        assert len(linear_dims) == 3,             "`linear_dims` must be [in1, out1, out2]"
        assert linear_dims[0] == conv_channels[-1], \
            "linear_dims[0] must equal last conv out-channels"

        # ---------- Convolution blocks ----------
        self.conv_blocks = nn.ModuleList()
        in_c = 2                                         # input: (B,2,10,16)
        for i, out_c in enumerate(conv_channels):
            if i == 0:
                self.conv_blocks.append(ConvMicroBlock(in_c, out_c))
            else:
                self.conv_blocks.append(SlimmableConvMicroBlock(in_c, out_c))
            in_c = out_c

        # Gate가 첫 블록의 GAP vector를 보니까 ↓
        self.depth_gate_conv = DepthGateConv(in_channels=conv_channels[0], max_depth=len(conv_channels))
        self.conv_gap = nn.AdaptiveAvgPool2d((1, 1))

        # ---------- Linear blocks ----------
        lin_pairs = list(zip(linear_dims[:-1], linear_dims[1:]))  # [(in1,out1), (out1,out2)]
        self.linear_blocks = nn.ModuleList([
            LinearBlockWithDynamicGate(in_feat, out_feat,
                                       (number_of_blocks_1, number_of_blocks_2)[i],
                                       num_steps=args.num_steps)
            for i, (in_feat, out_feat) in enumerate(lin_pairs)
        ])

        self.depth_gate = DepthGate(in_feat=linear_dims[1], max_depth=len(self.linear_blocks))

        # ---------- Heads ----------
        self.exit1_head = nn.Linear(linear_dims[1], 1)
        self.fc_pred    = nn.Linear(linear_dims[-1], 1)

        # Feature-hint projector
        self.proj_feat = nn.Linear(linear_dims[-1], linear_dims[-1], bias=False)
        with torch.no_grad():
            self.proj_feat.weight.copy_(torch.eye(linear_dims[-1]))

    # ------------------------------------------------------------
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 2, 10, 16)

        # ① Conv stage -------------------------------------------------
        feat = x
        for blk in self.conv_blocks:
            feat = blk(feat)

        gap_feat = self.conv_gap(feat).view(B, -1)  # (B, conv_channels[-1])

        # ② Linear stage ----------------------------------------------
        x1 = self.linear_blocks[0](gap_feat)              # first linear
        y_exit1 = self.exit1_head(x1)

        # y_stack
        feats = [x1]  # depth 1
        for i in range(1, len(self.linear_blocks)):  # depth 2…N
            feats.append(self.linear_blocks[i](feats[-1]))
        feat_stack = torch.stack(feats, dim=1)  # (B, N, hidden)
        y_stack = self.fc_pred(feat_stack).squeeze(-1)  # (B, N)

        depth_l, gate_probs, gate_logits = self.depth_gate(x1)

        out_lin = x1.clone()
        sel_l = (depth_l == 2).nonzero(as_tuple=True)[0]
        if sel_l.numel():
            out_lin[sel_l] = self.linear_blocks[1](out_lin[sel_l])

        y_final = self.fc_pred(out_lin)

        # width logits: conv_blocks[1:] 에만 있음
        width_logits = [blk.core[0].last_width_logits for blk in self.conv_blocks[1:]]

        # expert logits: 모든 linear_blocks
        expert_logits = [lb.last_expert_logits for lb in self.linear_blocks]

        return y_final, y_exit1, depth_l, y_stack, x1, out_lin, gate_logits, \
            width_logits, expert_logits # depth_c 미사용이면 None

tau0, tau_min, tau_gamma = 5.0, 0.3, 0.96

def gradnorm_update(losses, params, eta=1.5):
    # 처음 1iteration 때 losses_i(0) 저장 → 매 iter 갱신
    if not isinstance(params, (list, tuple)):
        params = list(params)
    if len(params) == 0:
        raise ValueError("Parameter list is empty!")

    if not hasattr(gradnorm_update, "init_losses"):
        gradnorm_update.init_losses = [l.detach() for l in losses]
        gradnorm_update.ws = [1.0]*len(losses)

    # 각 브랜치 그래드 L2-norm
    G = []
    for l in losses:
        grads = torch.autograd.grad(l, params, retain_graph=True, create_graph=True, allow_unused=True)
        valid = [g.norm() for g in grads if g is not None]
        norm = torch.stack(valid).mean() if valid else l.new_tensor(0.)
        G.append(norm)

    # target
    G_mean = torch.stack(G).mean().detach()
    # 새로운 가중치
    ws = []
    for i,(w0,L0,g) in enumerate(zip(gradnorm_update.ws, gradnorm_update.init_losses, G)):
        r   = (losses[i]/L0).detach()
        ws.append((w0 * (r**eta) * (G_mean/g).detach()).clamp(min=1e-3))

    # normalize
    s = sum(ws); ws = [w/s for w in ws]
    gradnorm_update.ws = ws

    return ws

def compute_layer_ops(model: nn.Module, input_size: tuple):
    """
    Automatically computes the number of operations (MACs) per layer using formulas from the paper:
    - Conv2d: OPS = Cin * Cout * kH * kW * Hout * Wout
    - Linear: OPS = in_features * out_features

    model: Your PyTorch nn.Module
    input_size: Tuple specifying (batch_size, channels, height, width) for an example input

    Returns a dict mapping layer names to their OPS count.
    """
    ops = {}
    hooks = []

    def conv_hook(self, input, output):
        Cin = input[0].shape[1]
        Cout = output.shape[1]
        Hout, Wout = output.shape[2], output.shape[3]
        kH, kW = self.kernel_size
        layer_ops = Cin * Cout * kH * kW * Hout * Wout
        ops[self] = layer_ops

    def linear_hook(self, input, output):
        Cin = input[0].shape[1]
        Cout = output.shape[1]
        layer_ops = Cin * Cout
        ops[self] = layer_ops

    # Register hooks on each Conv2d and Linear
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append((name, module.register_forward_hook(conv_hook)))
        elif isinstance(module, nn.Linear):
            hooks.append((name, module.register_forward_hook(linear_hook)))

    # Run a forward pass with dummy input
    model.eval()
    with torch.no_grad():
        model(torch.randn(input_size, device=next(model.parameters()).device))

    # Clean up hooks and build result dict
    result = {}
    for name, hook in hooks:
        module = dict(model.named_modules())[name]
        result[name] = ops.get(module, 0)
        hook.remove()

    return result

def build_gate_ops_tables(net, layer_ops):
    """
    layer_ops : compute_layer_ops() 로 얻은 {모듈경로: MACs} 딕셔너리
    net       : 학습에 사용할 모델
    반환값     : {
        "width_ops" : [MACs_per_width],
        "expert_ops_list" : [[blk0_e0, …],  [blk1_e0, …],  …],
        "depth_ops" : [MACs_depth1, MACs_depth2, …],
    }
    ------------------------------------------------------------------
    * 폭(width)  : conv_blocks[1:], pointwise 레이어 기준
    * 익스퍼트   : 각 LinearBlock 안의 dnn_experts.*, snn_experts.* 중
                  “*.0” (Linear/Conv 본체) 만 집계
    * 깊이(depth): early-exit 레벨별 누적 MACs (예시는 exit1 / full)
    """

    # ========== ① WIDTH 게이트 ==========
    width_ops = []
    for idx, blk in enumerate(net.conv_blocks[1:], start=1):  # conv_blocks[1:] 만 dynamic
        # 해당 블록의 pointwise 레이어 이름을 찾아 layer_ops에서 MACs 추출
        pattern = fr"conv_blocks\.{idx}\.core\.0\.pointwise"
        macs = next(m for n, m in layer_ops.items() if re.fullmatch(pattern, n))
        # 폭 게이트는 ratio(0.25/0.5/…)대로 채널 수만 달라지므로
        # base MACs × ratio 로 간단히 추정
        ratios = (0.25, 0.5, 0.75, 1.0)
        width_ops = [int(macs * r) for r in ratios]
        break  # 모두 동일 base라 한 번만 계산

    # ========== ② EXPERT 게이트 ==========
    expert_ops_list = []
    for i, lb in enumerate(net.linear_blocks):
        # dnn_experts.*.0     또는 snn_experts.*.0  만 집계
        pat = fr"linear_blocks\.{i}\.(dnn_experts|snn_experts)\.\d+\.0$"
        ops_this_blk = [
            macs for name, macs in layer_ops.items()
            if re.fullmatch(pat, name)
        ]
        expert_ops_list.append(sorted(ops_this_blk))  # K_i 길이 = 해당 블록 전문가 수

    # ========== ③ DEPTH 게이트 ==========
    #  예시 : exit1 = 폭블록[1] + expert블록[0]
    #         exit2 = 전체 네트(=full)   ← 수정 가능
    depth1_macs = 0
    #  conv_blocks[1] 전체 MAC 합산
    for name, macs in layer_ops.items():
        if name.startswith("conv_blocks.1."):
            depth1_macs += macs
    #  linear_blocks.0.* 전체 MAC 합산
    for name, macs in layer_ops.items():
        if name.startswith("linear_blocks.0."):
            depth1_macs += macs

    # full 모델 MACs
    full_macs = sum(layer_ops.values())
    depth_ops = [depth1_macs, full_macs]

    return {
        "width_ops": width_ops,
        "expert_ops_list": expert_ops_list,
        "depth_ops": depth_ops,
    }

def compute_expected_energy(gate_logits, ops_MACs, E_MAC=E_MAC, E_AC=E_AC, tau=1.0):
    # Gumbel-Softmax로 soft 확률
    probs = F.gumbel_softmax(gate_logits, tau=tau, hard=False)   # [B, K]
    energy_costs = gate_logits.new_tensor(ops_MACs) * (E_MAC + E_AC)  # [K] * pJ
    #   MACs ≈ ACs 라서  (MAC+AC) × pJ  로 단순화
    #   더 정밀하게 하려면   MACs*E_MAC + MACs*E_AC  따로 계산해도 OK
    E_batch_pJ = (probs * energy_costs).sum(dim=1).mean()        # scalar (pJ)

    return E_batch_pJ * 1e-12

if __name__ == "__main__":
    # Define loss and optimizer
    net = Net().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch_optim.Lookahead(optim.RAdam(net.parameters(), lr=learning_rate))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                     patience=args.schedular_patience)

    layer_ops = compute_layer_ops(net, input_size=(1, 2, 16, 10))

    expert_names = []
    for i, _ in enumerate(net.linear_blocks):
        pat = fr"linear_blocks\.{i}\.(dnn_experts|snn_experts)\.\d+\.0$"
        expert_names += [n for n in layer_ops if re.fullmatch(pat, n)]

    expert_energy_pJ = []
    for n in expert_names:
        macs = layer_ops[n]  # Conv/Linear → AC ≃ MAC
        if "snn_experts" in n:  # ▸ SNN  : AC만 × sparsity
            cost = macs * E_AC * SPIKE_RATE_AVG * T_STEPS
        else:  # ▸ DNN  : MAC+AC
            cost = macs * (E_MAC + E_AC)  # = macs × 3.2 pJ
        expert_energy_pJ.append(cost)

    gate_ops = build_gate_ops_tables(net, layer_ops)

    width_ops_MACs = gate_ops["width_ops"]  # [K_w]
    expert_ops_list = gate_ops["expert_ops_list"]  # [[K_e0], [K_e1], …]
    depth_ops_MACs = gate_ops["depth_ops"]  # [K_d]

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

            y_final, y_exit1, depth, y_stack, f_exit, f_final, gate_logits, width_logits, expert_logits = net(inputs)

            # Compute expected energy cost
            # ----- 폭 게이트 -----
            E_width_J = torch.stack([
                compute_expected_energy(logit,
                                        width_ops_MACs,  # 길이 K_w
                                        tau=blk.core[0].tau.item())
                for logit, blk in zip(width_logits, net.conv_blocks[1:])
            ]).sum()

            # ----- 익스퍼트 게이트 -----
            E_expert_J = torch.stack([
                compute_expected_energy_precalc(
                    logit,
                    expert_energy_pJ,  # ← 위에서 만든 pJ 리스트
                    tau=lb.tau.item())
                for logit, lb in zip(expert_logits, net.linear_blocks)
            ]).sum()

            # ----- 깊이 게이트 -----
            E_depth_J = compute_expected_energy(gate_logits,
                                                depth_ops_MACs,  # 길이 K_d
                                                tau=net.depth_gate.tau.item())

            E_total_J = E_width_J + E_expert_J + E_depth_J

            # ── (1) Main & Aux RMSE ─────────────────
            L_main = torch.sqrt(loss_fn(y_final, labels) + 1e-8)
            L_aux = torch.sqrt(loss_fn(y_exit1, labels) + 1e-8)
            L_each = torch.sqrt(F.mse_loss(y_stack, labels.squeeze(1).unsqueeze(1).expand_as(y_stack), reduction='none') + 1e-8)  # 각 depth별 loss

            # ── (2) Gate-Aware KD (depth==1 샘플만) ─
            sel1 = (depth == 1).nonzero(as_tuple=True)[0]
            KD = torch.tensor(0., device=device)

            if sel1.numel():
                # soft targets with temperature T
                KD = args.beta * torch.sqrt(loss_fn(y_exit1[sel1], y_final[sel1].detach()) + 1e-8)

            feat_t = f_final.detach()  # (B,32)
            feat_s = net.proj_feat(f_exit)  # (B,32)
            L_feat = args.gamma * torch.sqrt(loss_fn(feat_s, feat_t)+ 1e-8)  # feature hint loss

            loss_list = [L_aux, KD, L_feat]
            weight_vec = gradnorm_update(loss_list, net.parameters(), eta=args.eta)
            alpha, beta, gamma = weight_vec  # replace args.*
            # alpha, beta, gamma = 1.0, 1.0, 1.0

            routing_label = torch.argmin(L_each, dim=1)
            routing_ce = F.cross_entropy(gate_logits, routing_label)

            loss = L_main + alpha * L_aux + beta * KD + gamma * L_feat + args.delta * routing_ce + args.epsilon * E_total_J

            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.gradient_max_norm)
            optimizer.step()

            sched_tau = max(tau_min, tau0 * (tau_gamma ** epoch))
            anneal_and_clamp_tau(net, sched_tau)

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

                outputs = net(inputs)[0]

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
                    torch.save(net, "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/complete/cfo_scnn_wireless.pt")
                    break

    # torch.save(net, "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/cfo_scnn_wireless.pt")
