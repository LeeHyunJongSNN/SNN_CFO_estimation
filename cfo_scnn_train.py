"""Train the SDyNN CFO estimator.

This revision synchronizes the implementation with the manuscript in four
important places:
  1. convolutional early exit is part of the actual forward path;
  2. Gumbel-Softmax is used only during training, while evaluation uses a
     deterministic argmax;
  3. both SNN and DNN experts are evaluated during training to construct the
     minimum-error routing label (class 0=SNN, class 1=DNN);
  4. operation/FLOP accounting is analytical, group-aware, and includes the
     functional slimmable pointwise convolution and routing overhead.

The checkpoint format is state-dict based. Models trained with the older code
must be retrained because the old implementation did not execute Conv-EE and
did not train the MoE gate with minimum-error expert labels.
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import detrend
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from spikingjelly.activation_based import functional, neuron, surrogate


# ---------------------------------------------------------------------------
# Defaults and physical constants
# ---------------------------------------------------------------------------
DEFAULT_CONV_CHANNELS: Tuple[int, int] = (64, 64)
DEFAULT_LINEAR_DIMS: Tuple[int, int, int] = (64, 32, 32)
DEFAULT_WIDTHS: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
INPUT_SHAPE: Tuple[int, int, int] = (2, 10, 16)
INPUT_IQ_SAMPLES = 160

E_MAC_PJ = 3.1   # multiplication energy used by the manuscript
E_AC_PJ = 0.1    # accumulation energy used by the manuscript
SPIKE_RATE_AVG = 0.18
CHECKPOINT_VERSION = 2


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid Boolean value: {value!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--cutout", type=str2bool, default=False)
    parser.add_argument("--auto", type=str2bool, default=False)
    parser.add_argument("--num_lost", type=int, default=1)  # 1 ~ 5
    parser.add_argument("--conv_channels", type=int, nargs="+", default=list(DEFAULT_CONV_CHANNELS))
    parser.add_argument("--linear_dims", type=int, nargs="+", default=list(DEFAULT_LINEAR_DIMS))
    parser.add_argument("--num_blocks_1", type=int, default=2)
    parser.add_argument("--num_blocks_2", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--schedular_patience", type=int, default=2)
    parser.add_argument("--gradient_max_norm", type=float, default=5.0)
    parser.add_argument("--early_stop", type=str2bool, default=True)
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--spare_gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--train_file",
        type=str,
        default=(
            "/home/leehyunjong/Wi-Fi_Preambles/stfcfo/802.11ax_synth_changing/"
            "WiFi_20MHz_L-STF_ax_cfo_rapid_train.txt"
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/complete/"
            "cfo_scnn_wireless_ax_changing.pt"
        ),
    )
    return parser


def configure_device(gpu: bool, spare_gpu: int, seed: int) -> torch.device:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if spare_gpu != 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spare_gpu)

    use_cuda = gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(max(1, cpu_count - 1))
    return device


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------
def _select_gate(
    logits: torch.Tensor,
    tau: torch.Tensor | float,
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return one-hot gate and class indices.

    Training uses hard straight-through Gumbel-Softmax. Evaluation removes
    Gumbel noise and uses a deterministic argmax, as stated in the manuscript.
    """

    if training:
        tau_tensor = torch.as_tensor(tau, device=logits.device, dtype=logits.dtype).clamp_min(1e-4)
        gate = F.gumbel_softmax(logits, tau=tau_tensor, hard=True, dim=-1)
        index = gate.argmax(dim=-1)
    else:
        index = logits.argmax(dim=-1)
        gate = F.one_hot(index, num_classes=logits.shape[-1]).to(logits.dtype)
    return gate, index


def _relaxed_probabilities(logits: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
    """Deterministic relaxed routing probabilities for expected-energy loss."""

    tau_tensor = torch.as_tensor(tau, device=logits.device, dtype=logits.dtype).clamp_min(1e-4)
    return F.softmax(logits / tau_tensor, dim=-1)


@torch.no_grad()
def anneal_and_clamp_tau(model: nn.Module, scheduled_tau: float, tau_max: Optional[float] = None) -> None:
    for module in model.modules():
        if hasattr(module, "tau"):
            if tau_max is None:
                module.tau.data.clamp_(min=scheduled_tau)
            else:
                module.tau.data.clamp_(min=scheduled_tau, max=tau_max)


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------
class SEBlock2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.avg_pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class DSConv2dSE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        reduction: int = 4,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.se = SEBlock2d(out_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        return self.se(out)


class SlimmableDSConv2dSE(nn.Module):
    """Sample-wise slimmable DPSC block with switchable BN.

    The gate chooses one width per sample. During training, hard straight-
    through Gumbel routing is used. During evaluation, the choice is a
    deterministic argmax. Only the selected width is executed for each sample,
    and only that width's BN statistics are updated.
    """

    def __init__(
        self,
        in_channels: int,
        max_out: int,
        kernel_size: int = 3,
        widths: Sequence[float] = DEFAULT_WIDTHS,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        reduction: int = 4,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.max_out = int(max_out)
        self.widths = tuple(float(v) for v in widths)
        self.keep_channels = tuple(int(round(self.max_out * v)) for v in self.widths)
        if len(set(self.keep_channels)) != len(self.keep_channels):
            raise ValueError(f"Width multipliers produce duplicate channel counts: {self.keep_channels}")

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, max_out, kernel_size=1, bias=bias)
        self.bn_list = nn.ModuleList([nn.BatchNorm2d(k) for k in self.keep_channels])
        self.relu = nn.ReLU(inplace=False)
        self.se_list = nn.ModuleList([SEBlock2d(k, reduction) for k in self.keep_channels])
        self.gate_fc = nn.Linear(in_channels, len(self.widths))
        self.tau = nn.Parameter(torch.tensor(1.0))

        # This layer exists in the original implementation; it is retained and
        # explicitly counted by the revised FLOP profiler.
        self.up = nn.Conv2d(max_out, max_out, kernel_size=1, bias=False)
        with torch.no_grad():
            eye = torch.eye(max_out)
            self.up.weight.copy_(eye.view(max_out, max_out, 1, 1))

        self.last_width_logits: Optional[torch.Tensor] = None
        self.last_width_index: Optional[torch.Tensor] = None

    def _forward_selected_width(self, depthwise_out: torch.Tensor, width_index: int) -> torch.Tensor:
        keep = self.keep_channels[width_index]
        weight = self.pointwise.weight[:keep]
        bias = self.pointwise.bias[:keep] if self.pointwise.bias is not None else None
        out = F.conv2d(depthwise_out, weight, bias, stride=1, padding=0)
        out = self.bn_list[width_index](out)
        out = self.relu(out)
        out = self.se_list[width_index](out)
        if keep < self.max_out:
            pad = out.new_zeros(out.shape[0], self.max_out - keep, *out.shape[2:])
            out = torch.cat((out, pad), dim=1)
        return self.up(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depthwise_out = self.depthwise(x)
        pooled = F.adaptive_avg_pool2d(depthwise_out, 1).flatten(1)
        logits = self.gate_fc(pooled)
        gate, width_index = _select_gate(logits, self.tau, self.training)

        out = depthwise_out.new_zeros(
            depthwise_out.shape[0], self.max_out, depthwise_out.shape[2], depthwise_out.shape[3]
        )
        for k in range(len(self.widths)):
            selected = (width_index == k).nonzero(as_tuple=True)[0]
            if selected.numel() == 0:
                continue
            path = self._forward_selected_width(depthwise_out[selected], k)
            # Forward value is unchanged (the selected hard gate is one), while
            # the straight-through gate supplies a gradient to the width router.
            scale = gate[selected, k].view(-1, 1, 1, 1)
            out[selected] = path * scale

        self.last_width_logits = logits
        self.last_width_index = width_index
        return out


class ConvMicroBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.core = nn.Sequential(
            DSConv2dSE(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


class SlimmableConvMicroBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.core = nn.Sequential(
            SlimmableDSConv2dSE(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


class DepthGateConv(nn.Module):
    def __init__(self, in_channels: int, max_depth: int = 2, init_tau: float = 1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        self.fc = nn.Linear(in_channels, max_depth)

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.fc(feature)
        gate, index = _select_gate(logits, self.tau, self.training)
        return index + 1, gate, logits


class DepthGate(nn.Module):
    def __init__(self, in_features: int, max_depth: int = 2, init_tau: float = 1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        self.fc = nn.Linear(in_features, max_depth)

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.fc(feature)
        gate, index = _select_gate(logits, self.tau, self.training)
        return index + 1, gate, logits


class LinearBlockWithDynamicGate(nn.Module):
    """One SNN expert and one DNN expert with hard input-dependent routing.

    Gate class order is fixed to [SNN, DNN], matching the manuscript's routing
    label definition r=0 for SNN and r=1 for DNN.
    """

    def __init__(self, in_features: int, out_features: int, num_steps: int, init_tau: float = 1.0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_steps = int(num_steps)
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        self.dnn_expert = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())
        self.snn_expert = nn.Sequential(
            nn.Linear(in_features, out_features),
            neuron.IFNode(
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
            ),
        )
        self.gate_linear = nn.Linear(in_features, 2)  # class 0=SNN, class 1=DNN
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))

        self.last_expert_logits: Optional[torch.Tensor] = None
        self.last_expert_index: Optional[torch.Tensor] = None
        self.last_candidate_outputs: Optional[torch.Tensor] = None

    def _run_snn(self, x: torch.Tensor) -> torch.Tensor:
        # Membrane state is preserved across the configured time steps and is
        # reset only before/after the sample batch, matching multi-step SNN use.
        functional.reset_net(self.snn_expert)
        spike_sum: Optional[torch.Tensor] = None
        for _ in range(self.num_steps):
            current = self.snn_expert(x)
            spike_sum = current if spike_sum is None else spike_sum + current
        functional.reset_net(self.snn_expert)
        assert spike_sum is not None
        return spike_sum / float(self.num_steps)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        logits = self.gate_linear(x)
        gate, expert_index = _select_gate(logits, self.tau, self.training)

        if self.training:
            # Both experts are evaluated for every training sample. This is
            # required to construct the minimum-error routing label.
            snn_out = self._run_snn(x)
            dnn_out = self.dnn_expert(x)
            candidate_outputs = torch.stack((snn_out, dnn_out), dim=1)  # [B,2,D]
            out = (candidate_outputs * gate.unsqueeze(-1)).sum(dim=1)
        else:
            candidate_outputs = None
            out = x.new_zeros(x.shape[0], self.out_features)

            selected_snn = (expert_index == 0).nonzero(as_tuple=True)[0]
            if selected_snn.numel() > 0:
                out[selected_snn] = self._run_snn(x[selected_snn])

            selected_dnn = (expert_index == 1).nonzero(as_tuple=True)[0]
            if selected_dnn.numel() > 0:
                out[selected_dnn] = self.dnn_expert(x[selected_dnn])

        self.last_expert_logits = logits
        self.last_expert_index = expert_index
        self.last_candidate_outputs = candidate_outputs
        return out, logits, candidate_outputs, expert_index


# ---------------------------------------------------------------------------
# Model output and complete SDyNN
# ---------------------------------------------------------------------------
class SDyNNOutput(NamedTuple):
    prediction: torch.Tensor
    exit_prediction: torch.Tensor
    conv_depth: torch.Tensor
    linear_depth: torch.Tensor
    depth_predictions: torch.Tensor
    exit_feature: torch.Tensor
    full_feature: torch.Tensor
    conv_depth_logits: torch.Tensor
    linear_depth_logits: torch.Tensor
    width_logits: List[torch.Tensor]
    expert_logits: List[torch.Tensor]
    expert_candidate_predictions: List[Optional[torch.Tensor]]
    route_info: Dict[str, Any]


class Net(nn.Module):
    def __init__(
        self,
        conv_channels: Sequence[int] = DEFAULT_CONV_CHANNELS,
        linear_dims: Sequence[int] = DEFAULT_LINEAR_DIMS,
        number_of_blocks_1: int = 2,
        number_of_blocks_2: int = 2,
        num_steps: int = 2,
    ):
        super().__init__()
        conv_channels = tuple(int(v) for v in conv_channels)
        linear_dims = tuple(int(v) for v in linear_dims)

        if len(conv_channels) != 2:
            raise ValueError("The manuscript architecture requires exactly two convolution blocks.")
        if len(linear_dims) != 3:
            raise ValueError("linear_dims must be [input, hidden1, hidden2].")
        if linear_dims[0] != conv_channels[-1]:
            raise ValueError("linear_dims[0] must equal the convolution output channel count.")
        if number_of_blocks_1 != 2 or number_of_blocks_2 != 2:
            raise ValueError("Each linear block must contain exactly one SNN and one DNN expert.")

        self.config = {
            "conv_channels": list(conv_channels),
            "linear_dims": list(linear_dims),
            "number_of_blocks_1": 2,
            "number_of_blocks_2": 2,
            "num_steps": int(num_steps),
        }
        self.conv_channels = conv_channels
        self.linear_dims = linear_dims
        self.num_steps = int(num_steps)

        self.conv_blocks = nn.ModuleList(
            [
                ConvMicroBlock(INPUT_SHAPE[0], conv_channels[0]),
                SlimmableConvMicroBlock(conv_channels[0], conv_channels[1]),
            ]
        )
        self.depth_gate_conv = DepthGateConv(conv_channels[0], max_depth=2)
        self.conv_gap = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_blocks = nn.ModuleList(
            [
                LinearBlockWithDynamicGate(linear_dims[0], linear_dims[1], num_steps=num_steps),
                LinearBlockWithDynamicGate(linear_dims[1], linear_dims[2], num_steps=num_steps),
            ]
        )
        self.depth_gate = DepthGate(linear_dims[1], max_depth=2)

        self.exit1_head = nn.Linear(linear_dims[1], 1)
        self.fc_pred = nn.Linear(linear_dims[2], 1)
        self.proj_feat = nn.Linear(linear_dims[1], linear_dims[2], bias=False)
        if linear_dims[1] == linear_dims[2]:
            with torch.no_grad():
                self.proj_feat.weight.copy_(torch.eye(linear_dims[2]))

        # Stored in checkpoints and used for test-set denormalization.
        self.y_min: Optional[float] = None
        self.y_max: Optional[float] = None

    @staticmethod
    def _predict_candidates(head: nn.Linear, candidate_features: torch.Tensor) -> torch.Tensor:
        b, k, d = candidate_features.shape
        return head(candidate_features.reshape(b * k, d)).reshape(b, k)

    def _forward_convolution(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        batch = x.shape[0]
        first_feature = self.conv_blocks[0](x)
        first_gap = self.conv_gap(first_feature).flatten(1)
        conv_depth, conv_gate, conv_logits = self.depth_gate_conv(first_gap)

        width_logits_full = first_gap.new_full((batch, len(DEFAULT_WIDTHS)), float("nan"))
        width_index_full = torch.full((batch,), -1, device=x.device, dtype=torch.long)

        if self.training:
            # Training evaluates both Conv-EE candidates so that the hard
            # straight-through gate receives task gradients. Inference below
            # executes only the selected path.
            second_feature = self.conv_blocks[1](first_feature)
            second_gap = self.conv_gap(second_feature).flatten(1)
            gap = first_gap * conv_gate[:, 0:1] + second_gap * conv_gate[:, 1:2]

            core = self.conv_blocks[1].core[0]
            assert core.last_width_logits is not None and core.last_width_index is not None
            width_logits_full = core.last_width_logits
            width_index_full = core.last_width_index
        else:
            gap = first_gap.clone()
            selected_full = (conv_depth == 2).nonzero(as_tuple=True)[0]
            if selected_full.numel() > 0:
                second_feature = self.conv_blocks[1](first_feature[selected_full])
                second_gap = self.conv_gap(second_feature).flatten(1)
                gap[selected_full] = second_gap

                core = self.conv_blocks[1].core[0]
                assert core.last_width_logits is not None and core.last_width_index is not None
                width_logits_full[selected_full] = core.last_width_logits
                width_index_full[selected_full] = core.last_width_index

        return gap, conv_depth, conv_gate, conv_logits, [width_logits_full], [width_index_full]

    def forward(self, x: torch.Tensor) -> SDyNNOutput:
        batch = x.shape[0]
        x = x.view(batch, *INPUT_SHAPE)

        gap, conv_depth, _, conv_logits, width_logits, width_indices = self._forward_convolution(x)

        # First linear MoE block is always reached.
        x1, expert_logits_1, candidates_1, expert_index_1 = self.linear_blocks[0](gap)
        linear_depth, linear_gate, linear_logits = self.depth_gate(x1)

        expert_logits_2_full = x1.new_full((batch, 2), float("nan"))
        expert_index_2_full = torch.full((batch,), -1, device=x.device, dtype=torch.long)

        if self.training:
            # Both linear depths are evaluated during training for the EE
            # supervision losses and the minimum-error label of block 2.
            y_exit = self.exit1_head(x1)
            x2, expert_logits_2, candidates_2, expert_index_2 = self.linear_blocks[1](x1)
            y_full = self.fc_pred(x2)
            y_final = y_exit * linear_gate[:, 0:1] + y_full * linear_gate[:, 1:2]
            final_feature = x2
            expert_logits_2_full = expert_logits_2
            expert_index_2_full = expert_index_2

            assert candidates_1 is not None and candidates_2 is not None
            candidate_pred_1 = self._predict_candidates(self.exit1_head, candidates_1)
            candidate_pred_2 = self._predict_candidates(self.fc_pred, candidates_2)
            candidate_predictions: List[Optional[torch.Tensor]] = [candidate_pred_1, candidate_pred_2]
            depth_predictions = torch.cat((y_exit, y_full), dim=1)
        else:
            # True early exit: an intermediate predictor is evaluated only for
            # depth-1 samples, while block 2 and the final predictor are
            # evaluated only for depth-2 samples.
            y_exit = x1.new_full((batch, 1), float("nan"))
            y_full_placeholder = x1.new_full((batch, 1), float("nan"))
            y_final = x1.new_zeros((batch, 1))
            final_feature = x1.clone()

            selected_exit = (linear_depth == 1).nonzero(as_tuple=True)[0]
            if selected_exit.numel() > 0:
                exit_value = self.exit1_head(x1[selected_exit])
                y_exit[selected_exit] = exit_value
                y_final[selected_exit] = exit_value

            selected_full = (linear_depth == 2).nonzero(as_tuple=True)[0]
            if selected_full.numel() > 0:
                x2, logits_2, _, index_2 = self.linear_blocks[1](x1[selected_full])
                y_full = self.fc_pred(x2)
                y_final[selected_full] = y_full
                final_feature[selected_full] = x2
                y_full_placeholder[selected_full] = y_full
                expert_logits_2_full[selected_full] = logits_2
                expert_index_2_full[selected_full] = index_2

            candidate_predictions = [None, None]
            depth_predictions = torch.cat((y_exit, y_full_placeholder), dim=1)

        route_info: Dict[str, Any] = {
            "conv_depth": conv_depth,
            "linear_depth": linear_depth,
            "width_indices": width_indices,
            "expert_indices": [expert_index_1, expert_index_2_full],
        }

        return SDyNNOutput(
            prediction=y_final,
            exit_prediction=y_exit,
            conv_depth=conv_depth,
            linear_depth=linear_depth,
            depth_predictions=depth_predictions,
            exit_feature=x1,
            full_feature=final_feature,
            conv_depth_logits=conv_logits,
            linear_depth_logits=linear_logits,
            width_logits=width_logits,
            expert_logits=[expert_logits_1, expert_logits_2_full],
            expert_candidate_predictions=candidate_predictions,
            route_info=route_info,
        )


# ---------------------------------------------------------------------------
# Correct analytical operation/FLOP accounting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperationCost:
    macs: float = 0.0
    acs: float = 0.0
    other_ops: float = 0.0

    def __add__(self, other: "OperationCost") -> "OperationCost":
        return OperationCost(
            self.macs + other.macs,
            self.acs + other.acs,
            self.other_ops + other.other_ops,
        )

    def __mul__(self, scalar: float) -> "OperationCost":
        return OperationCost(self.macs * scalar, self.acs * scalar, self.other_ops * scalar)

    __rmul__ = __mul__

    @property
    def flops(self) -> float:
        # One dense MAC contains one multiplication and one accumulation.
        return 2.0 * self.macs + self.acs + self.other_ops

    @property
    def energy_pj(self) -> float:
        return self.macs * (E_MAC_PJ + E_AC_PJ) + self.acs * E_AC_PJ


def _conv2d_macs(
    in_channels: int,
    out_channels: int,
    kernel_size: int | Tuple[int, int],
    out_h: int,
    out_w: int,
    groups: int = 1,
) -> int:
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size
    if in_channels % groups != 0:
        raise ValueError("in_channels must be divisible by groups")
    return int(out_h * out_w * out_channels * (in_channels // groups) * kh * kw)


def _linear_macs(in_features: int, out_features: int) -> int:
    return int(in_features * out_features)


def _gap_ops(channels: int, height: int, width: int) -> int:
    # Additions needed for global averaging; division is one further operation
    # per channel. This is routing/auxiliary overhead, not a MAC.
    return int(channels * max(0, height * width - 1) + channels)


def _se_cost(channels: int, height: int, width: int, reduction: int = 4) -> OperationCost:
    hidden = max(1, channels // reduction)
    fc_macs = _linear_macs(channels, hidden) + _linear_macs(hidden, channels)
    scale_multiplies = channels * height * width
    return OperationCost(macs=fc_macs + scale_multiplies, other_ops=_gap_ops(channels, height, width))


def build_operation_costs(net: Net, spike_rate: float = SPIKE_RATE_AVG) -> Dict[str, Any]:
    """Build route-component costs without stochastic hooks.

    This fixes the previous counter by handling depthwise ``groups`` correctly,
    counting the functional sliced pointwise convolution, and keeping repeated
    route components separate instead of overwriting hook results.
    """

    c0, c1 = net.conv_channels
    _, h0, w0 = INPUT_SHAPE
    h1, w1 = h0 // 2, w0 // 2
    h2, w2 = h1 // 2, w1 // 2

    conv1 = OperationCost(
        macs=(
            _conv2d_macs(INPUT_SHAPE[0], INPUT_SHAPE[0], 3, h0, w0, groups=INPUT_SHAPE[0])
            + _conv2d_macs(INPUT_SHAPE[0], c0, 1, h0, w0)
        )
    ) + _se_cost(c0, h0, w0)

    conv_depth_gate = OperationCost(
        macs=_linear_macs(c0, 2),
        other_ops=_gap_ops(c0, h1, w1),
    )

    width_gate = OperationCost(
        macs=_linear_macs(c0, len(DEFAULT_WIDTHS)),
        other_ops=_gap_ops(c0, h1, w1),
    )

    conv2_by_width: List[OperationCost] = []
    for keep in net.conv_blocks[1].core[0].keep_channels:
        dynamic_conv = OperationCost(
            macs=(
                _conv2d_macs(c0, c0, 3, h1, w1, groups=c0)
                + _conv2d_macs(c0, keep, 1, h1, w1)
                + _conv2d_macs(c1, c1, 1, h1, w1)  # retained learned up-projection
            ),
            other_ops=_gap_ops(c1, h2, w2),
        )
        conv2_by_width.append(dynamic_conv + _se_cost(keep, h1, w1))

    linear_experts: List[List[OperationCost]] = []
    expert_gates: List[OperationCost] = []
    for block in net.linear_blocks:
        dense_macs = _linear_macs(block.in_features, block.out_features)
        snn_acs = dense_macs * spike_rate * block.num_steps
        linear_experts.append(
            [
                OperationCost(acs=snn_acs),        # class 0: SNN
                OperationCost(macs=dense_macs),    # class 1: DNN
            ]
        )
        expert_gates.append(OperationCost(macs=_linear_macs(block.in_features, 2)))

    linear_depth_gate = OperationCost(macs=_linear_macs(net.linear_dims[1], 2))
    exit_head = OperationCost(macs=_linear_macs(net.linear_dims[1], 1))
    final_head = OperationCost(macs=_linear_macs(net.linear_dims[2], 1))

    return {
        "conv1": conv1,
        "conv_depth_gate": conv_depth_gate,
        "width_gate": width_gate,
        "conv2_by_width": conv2_by_width,
        "expert_gates": expert_gates,
        "linear_experts": linear_experts,
        "linear_depth_gate": linear_depth_gate,
        "exit_head": exit_head,
        "final_head": final_head,
    }


def compute_layer_ops(model: Net, input_size: Tuple[int, int, int, int] = (1, 2, 10, 16)) -> Dict[str, int]:
    """Compatibility wrapper returning analytical MAC counts by component.

    ``input_size`` is validated but route costs are derived analytically from
    the manuscript architecture. Values are MAC/AC-equivalent operation counts;
    use :func:`summarize_flops` for conventional FLOPs.
    """

    if tuple(input_size[1:]) != INPUT_SHAPE:
        raise ValueError(f"Expected input shape (B,{INPUT_SHAPE}), got {input_size}")
    costs = build_operation_costs(model)
    result: Dict[str, int] = {
        "conv_blocks.0": int(round(costs["conv1"].macs + costs["conv1"].acs)),
        "depth_gate_conv": int(round(costs["conv_depth_gate"].macs + costs["conv_depth_gate"].other_ops)),
        "slimmable_width_gate": int(round(costs["width_gate"].macs + costs["width_gate"].other_ops)),
        "linear_depth_gate": int(round(costs["linear_depth_gate"].macs)),
        "exit_head": int(round(costs["exit_head"].macs)),
        "final_head": int(round(costs["final_head"].macs)),
    }
    for i, cost in enumerate(costs["conv2_by_width"]):
        keep = model.conv_blocks[1].core[0].keep_channels[i]
        result[f"conv_blocks.1.width_{keep}"] = int(round(cost.macs + cost.acs + cost.other_ops))
    for block_idx, (gate_cost, expert_costs) in enumerate(
        zip(costs["expert_gates"], costs["linear_experts"])
    ):
        result[f"linear_blocks.{block_idx}.gate"] = int(round(gate_cost.macs))
        result[f"linear_blocks.{block_idx}.snn"] = int(round(expert_costs[0].acs))
        result[f"linear_blocks.{block_idx}.dnn"] = int(round(expert_costs[1].macs))
    return result


def _full_reference_cost(costs: Dict[str, Any]) -> OperationCost:
    return (
        costs["conv1"]
        + costs["conv_depth_gate"]
        + costs["width_gate"]
        + costs["conv2_by_width"][-1]
        + costs["expert_gates"][0]
        + costs["linear_experts"][0][1]
        + costs["linear_depth_gate"]
        + costs["expert_gates"][1]
        + costs["linear_experts"][1][1]
        + costs["final_head"]
    )


def gate_only_full_path_cost(net: Net) -> OperationCost:
    costs = build_operation_costs(net)
    return (
        costs["conv_depth_gate"]
        + costs["width_gate"]
        + costs["expert_gates"][0]
        + costs["linear_depth_gate"]
        + costs["expert_gates"][1]
    )


def summarize_flops(net: Net) -> Dict[str, float]:
    costs = build_operation_costs(net)
    reference = _full_reference_cost(costs)
    gate_cost = gate_only_full_path_cost(net)
    return {
        "full_reference_flops": reference.flops,
        "full_path_gate_flops": gate_cost.flops,
        "gate_fraction_of_full_percent": 100.0 * gate_cost.flops / reference.flops,
        "ungated_to_gated_increase_percent": 100.0 * gate_cost.flops / (reference.flops - gate_cost.flops),
    }


def expected_energy_loss(output: SDyNNOutput, net: Net) -> torch.Tensor:
    """Equation-(8)--(10)-style normalized expected inference energy."""

    costs = build_operation_costs(net)
    device = output.prediction.device
    dtype = output.prediction.dtype

    p_conv = _relaxed_probabilities(output.conv_depth_logits, net.depth_gate_conv.tau)
    p_width = _relaxed_probabilities(output.width_logits[0], net.conv_blocks[1].core[0].tau)
    p_linear = _relaxed_probabilities(output.linear_depth_logits, net.depth_gate.tau)
    p_expert_1 = _relaxed_probabilities(output.expert_logits[0], net.linear_blocks[0].tau)
    p_expert_2 = _relaxed_probabilities(output.expert_logits[1], net.linear_blocks[1].tau)

    def energy_tensor(items: Sequence[OperationCost]) -> torch.Tensor:
        return torch.tensor([item.energy_pj for item in items], device=device, dtype=dtype)

    fixed = costs["conv1"].energy_pj + costs["conv_depth_gate"].energy_pj
    conv2_energy = costs["width_gate"].energy_pj + (p_width * energy_tensor(costs["conv2_by_width"])).sum(dim=1)

    linear1_energy = costs["expert_gates"][0].energy_pj + (
        p_expert_1 * energy_tensor(costs["linear_experts"][0])
    ).sum(dim=1)
    linear2_energy = costs["expert_gates"][1].energy_pj + (
        p_expert_2 * energy_tensor(costs["linear_experts"][1])
    ).sum(dim=1)

    head_energy = p_linear[:, 0] * costs["exit_head"].energy_pj + p_linear[:, 1] * (
        linear2_energy + costs["final_head"].energy_pj
    )

    expected_pj = (
        fixed
        + p_conv[:, 1] * conv2_energy
        + linear1_energy
        + costs["linear_depth_gate"].energy_pj
        + head_energy
    )

    reference_pj = _full_reference_cost(costs).energy_pj
    return (expected_pj / reference_pj).mean()




def estimate_inference_gate_flops(output: SDyNNOutput, net: Net) -> torch.Tensor:
    """Return the routing-network FLOPs actually executed for each sample."""

    costs = build_operation_costs(net)
    route = output.route_info
    conv_depth = route["conv_depth"].detach().cpu()
    linear_depth = route["linear_depth"].detach().cpu()

    values: List[float] = []
    for i in range(conv_depth.numel()):
        cost = costs["conv_depth_gate"] + costs["expert_gates"][0] + costs["linear_depth_gate"]
        if int(conv_depth[i]) == 2:
            cost = cost + costs["width_gate"]
        if int(linear_depth[i]) == 2:
            cost = cost + costs["expert_gates"][1]
        values.append(cost.flops)
    return torch.tensor(values, dtype=torch.float64)

def estimate_inference_flops(output: SDyNNOutput, net: Net) -> torch.Tensor:
    """Return actual route-dependent FLOPs for every sample in ``output``."""

    costs = build_operation_costs(net)
    route = output.route_info
    conv_depth = route["conv_depth"].detach().cpu()
    linear_depth = route["linear_depth"].detach().cpu()
    width_idx = route["width_indices"][0].detach().cpu()
    expert_1 = route["expert_indices"][0].detach().cpu()
    expert_2 = route["expert_indices"][1].detach().cpu()

    values: List[float] = []
    for i in range(conv_depth.numel()):
        cost = costs["conv1"] + costs["conv_depth_gate"]
        if int(conv_depth[i]) == 2:
            wi = int(width_idx[i])
            if wi < 0:
                raise RuntimeError("Conv depth 2 was selected but no width index was recorded.")
            cost = cost + costs["width_gate"] + costs["conv2_by_width"][wi]

        e1 = int(expert_1[i])
        if e1 not in (0, 1):
            raise RuntimeError(f"Invalid first expert index: {e1}")
        cost = cost + costs["expert_gates"][0] + costs["linear_experts"][0][e1]
        cost = cost + costs["linear_depth_gate"]

        if int(linear_depth[i]) == 1:
            cost = cost + costs["exit_head"]
        else:
            e2 = int(expert_2[i])
            if e2 not in (0, 1):
                raise RuntimeError("Linear depth 2 was selected but no second expert index was recorded.")
            cost = cost + costs["expert_gates"][1] + costs["linear_experts"][1][e2]
            cost = cost + costs["final_head"]
        values.append(cost.flops)
    return torch.tensor(values, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def minimum_error_expert_labels(
    candidate_predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Build training-only labels: 0=SNN, 1=DNN from the smaller CFO error."""

    if candidate_predictions.ndim != 2 or candidate_predictions.shape[1] != 2:
        raise ValueError("candidate_predictions must have shape [B,2] in [SNN,DNN] order")
    target = targets.view(-1, 1)
    error = (candidate_predictions.detach() - target).abs()
    return error.argmin(dim=1).long()


def binary_expert_routing_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy for P(DNN); class 0=SNN and class 1=DNN."""

    if logits.shape[-1] != 2:
        raise ValueError("Expert gate logits must have two classes [SNN,DNN].")
    dnn_logit = logits[:, 1] - logits[:, 0]
    return F.binary_cross_entropy_with_logits(dnn_logit, labels.to(logits.dtype))


def gradnorm_update(losses: Sequence[torch.Tensor], params: Sequence[nn.Parameter], eta: float = 1.5) -> List[torch.Tensor]:
    """Retain the original GradNorm-inspired multiplicative balancing rule."""

    params = list(params)
    if not params:
        raise ValueError("Parameter list is empty")

    if not hasattr(gradnorm_update, "init_losses"):
        gradnorm_update.init_losses = [loss.detach().clamp_min(1e-8) for loss in losses]
        gradnorm_update.weights = [1.0] * len(losses)

    norms: List[torch.Tensor] = []
    for loss in losses:
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )
        valid = [gradient.norm() for gradient in grads if gradient is not None]
        norms.append(torch.stack(valid).mean() if valid else loss.new_tensor(1e-8))

    mean_norm = torch.stack(norms).mean().detach()
    updated: List[torch.Tensor] = []
    for old_weight, initial_loss, loss, norm in zip(
        gradnorm_update.weights,
        gradnorm_update.init_losses,
        losses,
        norms,
    ):
        relative = (loss / initial_loss).detach().clamp_min(1e-8)
        updated.append(
            (old_weight * relative.pow(eta) * (mean_norm / norm.clamp_min(1e-8)).detach()).clamp(min=1e-3)
        )

    scale = sum(updated)
    normalized = [value / scale for value in updated]
    gradnorm_update.weights = normalized
    return normalized


# ---------------------------------------------------------------------------
# Data and checkpoint I/O
# ---------------------------------------------------------------------------
def preprocess_cfo_file(path: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    raw = np.loadtxt(path, dtype=str, delimiter="\t")
    np.random.shuffle(raw)
    raw = np.char.replace(raw, "i", "j").astype(np.complex64)

    signals: List[np.ndarray] = []
    labels: List[float] = []
    eps = np.finfo(np.float32).eps
    for line in raw:
        data = line[:INPUT_IQ_SAMPLES]
        label = float(np.real(line[-1]))
        centered = detrend(data - np.mean(data))
        real = np.real(centered).astype(np.float32)
        imag = np.imag(centered).astype(np.float32)
        real_rms = max(float(np.sqrt(np.mean(real ** 2))), eps)
        imag_rms = max(float(np.sqrt(np.mean(imag ** 2))), eps)
        signals.append(np.stack((real / real_rms, imag / imag_rms), axis=0))
        labels.append(label)

    x = np.stack(signals).astype(np.float32)
    y_raw = np.asarray(labels, dtype=np.float32)
    y_min = float(y_raw.min())
    y_max = float(y_raw.max())
    if not y_max > y_min:
        raise ValueError("CFO labels must span a nonzero range")
    y = ((y_raw - y_min) / (y_max - y_min)).reshape(-1, 1).astype(np.float32)
    return x, y, y_min, y_max


def apply_cutout(inputs: torch.Tensor, automatic: bool, num_lost: int) -> torch.Tensor:
    mask = torch.ones_like(inputs)
    for i in range(inputs.shape[0]):
        position = int(torch.randint(1, 6, (1,), device=inputs.device).item()) if automatic else int(num_lost)
        if position < 1 or position > 5:
            raise ValueError("num_lost must be in [1,5]")
        mask[i, :, : 32 * (position - 1)] = 0
    return inputs * mask


def save_checkpoint(path: str, net: Net, y_min: float, y_max: float, extra: Optional[Dict[str, Any]] = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format_version": CHECKPOINT_VERSION,
        "model_config": net.config,
        "model_state": net.state_dict(),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, target)


def load_checkpoint(path: str, device: torch.device | str) -> Tuple[Net, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise RuntimeError(
            "This script requires a version-2 state-dict checkpoint. The previous whole-model checkpoint "
            "was produced by an implementation without functional Conv-EE/minimum-error MoE labels and "
            "must be retrained with the synchronized training script."
        )
    net = Net(**checkpoint["model_config"]).to(device)
    net.load_state_dict(checkpoint["model_state"], strict=True)
    net.y_min = float(checkpoint["y_min"])
    net.y_max = float(checkpoint["y_max"])
    return net, checkpoint


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_arg_parser().parse_args()
    device = configure_device(args.gpu, args.spare_gpu, args.seed)

    x_np, y_np, y_min, y_max = preprocess_cfo_file(args.train_file)
    x = torch.tensor(x_np, device=device)
    y = torch.tensor(y_np, device=device)
    dataset = torch.utils.data.TensorDataset(x, y)

    generator = torch.Generator().manual_seed(args.seed)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    net = Net(
        conv_channels=args.conv_channels,
        linear_dims=args.linear_dims,
        number_of_blocks_1=args.num_blocks_1,
        number_of_blocks_2=args.num_blocks_2,
        num_steps=args.num_steps,
    ).to(device)
    net.y_min, net.y_max = y_min, y_max

    try:
        import torch_optimizer as torch_optim

        optimizer: optim.Optimizer = torch_optim.Lookahead(
            optim.RAdam(net.parameters(), lr=args.learning_rate)
        )
    except ImportError:
        print("[warning] torch_optimizer is unavailable; using torch.optim.RAdam without Lookahead.")
        optimizer = optim.RAdam(net.parameters(), lr=args.learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.schedular_patience,
    )
    mse = nn.MSELoss()

    flop_summary = summarize_flops(net)
    print(
        "Analytical full-reference FLOPs: "
        f"{flop_summary['full_reference_flops']:.0f}; full-path routing FLOPs: "
        f"{flop_summary['full_path_gate_flops']:.0f} "
        f"({flop_summary['gate_fraction_of_full_percent']:.3f}% of full reference)."
    )

    best_val_loss = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    tau0, tau_min, tau_gamma = 5.0, 0.3, 0.96

    # Reset persistent state if main() is invoked repeatedly in one process.
    for attr in ("init_losses", "weights"):
        if hasattr(gradnorm_update, attr):
            delattr(gradnorm_update, attr)

    for epoch in range(args.n_epochs):
        net.train()
        train_losses: List[float] = []
        routing_correct = [0, 0]
        routing_total = [0, 0]

        for inputs, labels in train_loader:
            if args.cutout:
                inputs = apply_cutout(inputs, args.auto, args.num_lost)

            optimizer.zero_grad(set_to_none=True)
            output = net(inputs)

            l_main = torch.sqrt(mse(output.prediction, labels) + 1e-8)
            l_ea = torch.sqrt(mse(output.exit_prediction, labels) + 1e-8)

            selected_exit = (output.linear_depth == 1).nonzero(as_tuple=True)[0]
            l_ef = output.exit_prediction.sum() * 0.0
            if selected_exit.numel() > 0:
                full_prediction = output.depth_predictions[:, 1:2]
                l_ef = torch.sqrt(
                    mse(
                        output.exit_prediction[selected_exit],
                        full_prediction[selected_exit].detach(),
                    )
                    + 1e-8
                )

            projected_exit = net.proj_feat(output.exit_feature)
            l_fc = torch.sqrt(mse(projected_exit, output.full_feature.detach()) + 1e-8)

            weights = gradnorm_update([l_ea, l_ef, l_fc], list(net.parameters()), eta=args.eta)
            l_exit = weights[0] * l_ea + weights[1] * l_ef + weights[2] * l_fc

            moe_losses: List[torch.Tensor] = []
            for block_idx, (candidate_pred, gate_logits) in enumerate(
                zip(output.expert_candidate_predictions, output.expert_logits)
            ):
                if candidate_pred is None:
                    raise RuntimeError("Training must return both expert predictions for routing labels.")
                route_label = minimum_error_expert_labels(candidate_pred, labels)
                moe_losses.append(binary_expert_routing_loss(gate_logits, route_label))
                predicted_route = gate_logits.detach().argmax(dim=1)
                routing_correct[block_idx] += int((predicted_route == route_label).sum().item())
                routing_total[block_idx] += int(route_label.numel())
            l_gate = torch.stack(moe_losses).mean()

            l_energy = expected_energy_loss(output, net)
            loss = l_main + l_exit + args.delta * l_gate + args.epsilon * l_energy
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss: main={l_main.item()}, exit={l_exit.item()}, "
                    f"gate={l_gate.item()}, energy={l_energy.item()}"
                )

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.gradient_max_norm)
            optimizer.step()
            functional.reset_net(net)
            train_losses.append(float(loss.item()))

        scheduled_tau = max(tau_min, tau0 * (tau_gamma ** epoch))
        anneal_and_clamp_tau(net, scheduled_tau)
        average_train_loss = float(np.mean(train_losses))

        net.eval()
        val_losses: List[float] = []
        val_maes: List[float] = []
        with torch.inference_mode():
            for inputs, labels in val_loader:
                if args.cutout:
                    inputs = apply_cutout(inputs, args.auto, args.num_lost)
                output = net(inputs)
                val_losses.append(float(torch.sqrt(mse(output.prediction, labels) + 1e-8).item()))
                val_maes.append(float(torch.mean(torch.abs(output.prediction - labels)).item()))
                functional.reset_net(net)

        average_val_loss = float(np.mean(val_losses))
        denormalized_val_mae = float(np.mean(val_maes)) * (y_max - y_min)
        scheduler.step(average_val_loss)
        route_acc = [
            (routing_correct[i] / routing_total[i]) if routing_total[i] else float("nan")
            for i in range(2)
        ]
        print(
            f"Epoch {epoch + 1}/{args.n_epochs} - Train loss: {average_train_loss:.5f} - "
            f"Val RMSE: {average_val_loss:.5f} - Val MAE: {denormalized_val_mae:.2f} Hz - "
            f"MoE route acc: block1={route_acc[0]:.3f}, block2={route_acc[1]:.3f}"
        )

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_state = copy.deepcopy(net.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(
                args.model_path,
                net,
                y_min,
                y_max,
                extra={"epoch": epoch + 1, "best_val_rmse": best_val_loss},
            )
        else:
            epochs_without_improvement += 1

        if args.early_stop and epochs_without_improvement >= args.es_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")
    net.load_state_dict(best_state)
    save_checkpoint(
        args.model_path,
        net,
        y_min,
        y_max,
        extra={"best_val_rmse": best_val_loss},
    )
    print(f"Saved synchronized SDyNN checkpoint to: {args.model_path}")


if __name__ == "__main__":
    main()
