"""Log SDyNN route selections and actual route-dependent FLOPs.

The script reads route metadata returned by the model itself. It no longer
re-implements the convolution path manually or double-counts the Conv-EE hook.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import detrend
import torch
from spikingjelly.activation_based import functional


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid Boolean value: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--cutout", type=str2bool, default=True)
    parser.add_argument("--auto", type=str2bool, default=True)
    parser.add_argument("--num_lost", type=int, default=1)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--spare_gpu", type=int, default=0)
    parser.add_argument("--shuffle_test", type=str2bool, default=False)
    parser.add_argument("--train_module_dir", type=str, default="/SNN/CFO/train")
    parser.add_argument(
        "--test_file",
        type=str,
        default=(
            "/home/leehyunjong/Wi-Fi_Preambles/stfcfo/wireless/"
            "WiFi_10MHz_Preambles_wireless_cfo_test_rician_18dB.txt"
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "/home/leehyunjong/PycharmProjects/Machine_Learning/SNN/CFO/models/incomplete/"
            "cfo_scnn_wireless.pt"
        ),
    )
    return parser


def configure_device(gpu: bool, spare_gpu: int) -> torch.device:
    if spare_gpu != 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spare_gpu)
    return torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")


def load_test_file(path: str, shuffle: bool) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, dtype=str, delimiter="\t")
    if shuffle:
        np.random.shuffle(raw)
    raw = np.char.replace(raw, "i", "j").astype(np.complex64)

    signals: List[np.ndarray] = []
    labels: List[float] = []
    eps = np.finfo(np.float32).eps
    for line in raw:
        data = line[:160]
        centered = detrend(data - np.mean(data))
        real = np.real(centered).astype(np.float32)
        imag = np.imag(centered).astype(np.float32)
        real_rms = max(float(np.sqrt(np.mean(real ** 2))), eps)
        imag_rms = max(float(np.sqrt(np.mean(imag ** 2))), eps)
        signals.append(np.stack((real / real_rms, imag / imag_rms), axis=0))
        labels.append(float(np.real(line[-1])))
    return np.stack(signals).astype(np.float32), np.asarray(labels, dtype=np.float32)


def print_counter(title: str, values: List[int], labels: Dict[int, str]) -> None:
    print(f"\n-- {title} --")
    counts = Counter(values)
    total = sum(counts.values())
    for key in sorted(counts):
        name = labels.get(key, str(key))
        proportion = counts[key] / total if total else float("nan")
        print(f"{name}: {counts[key]} ({proportion:.4f})")


def main() -> None:
    args = build_parser().parse_args()
    device = configure_device(args.gpu, args.spare_gpu)

    train_dir = str(Path(args.train_module_dir).resolve())
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    from cfo_scnn_train_v2 import (
        apply_cutout,
        estimate_inference_flops,
        estimate_inference_gate_flops,
        gate_only_full_path_cost,
        load_checkpoint,
        summarize_flops,
    )

    x_np, y_np = load_test_file(args.test_file, args.shuffle_test)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np).unsqueeze(1))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    net, checkpoint = load_checkpoint(args.model_path, device)
    net.eval()
    y_min = float(checkpoint["y_min"])
    y_max = float(checkpoint["y_max"])

    conv_depths: List[int] = []
    linear_depths: List[int] = []
    width_indices: List[int] = []
    expert_1: List[int] = []
    expert_2: List[int] = []
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    flops_all: List[torch.Tensor] = []
    gate_flops_all: List[torch.Tensor] = []
    route_keys: List[Tuple[int, int, int, int, int]] = []

    with torch.inference_mode():
        for inputs, target in loader:
            inputs = inputs.to(device)
            if args.cutout:
                inputs = apply_cutout(inputs, args.auto, args.num_lost)

            output = net(inputs)
            route = output.route_info
            conv_depths.extend(route["conv_depth"].cpu().tolist())
            linear_depths.extend(route["linear_depth"].cpu().tolist())
            width_indices.extend(route["width_indices"][0].cpu().tolist())
            expert_1.extend(route["expert_indices"][0].cpu().tolist())
            expert_2.extend(route["expert_indices"][1].cpu().tolist())

            prediction_hz = output.prediction * (y_max - y_min) + y_min
            predictions.append(prediction_hz.cpu().numpy())
            targets.append(target.numpy())
            batch_flops = estimate_inference_flops(output, net)
            batch_gate_flops = estimate_inference_gate_flops(output, net)
            flops_all.append(batch_flops)
            gate_flops_all.append(batch_gate_flops)

            batch_conv = route["conv_depth"].cpu().tolist()
            batch_width = route["width_indices"][0].cpu().tolist()
            batch_expert_1 = route["expert_indices"][0].cpu().tolist()
            batch_linear = route["linear_depth"].cpu().tolist()
            batch_expert_2 = route["expert_indices"][1].cpu().tolist()
            route_keys.extend(
                (int(cd), int(wi), int(e1), int(ld), int(e2))
                for cd, wi, e1, ld, e2 in zip(
                    batch_conv, batch_width, batch_expert_1, batch_linear, batch_expert_2
                )
            )
            functional.reset_net(net)

    prediction = np.concatenate(predictions, axis=0).reshape(-1)
    target = np.concatenate(targets, axis=0).reshape(-1)

    print_counter("Convolution-stage depth distribution", conv_depths, {1: "CD1 (early exit)", 2: "CD2 (full)"})
    print_counter("Linear-stage depth distribution", linear_depths, {1: "LD1 (early exit)", 2: "LD2 (full)"})

    reached_width = [value for value in width_indices if value >= 0]
    skipped_width = sum(value < 0 for value in width_indices)
    width_labels = {
        index: f"CF{channels}"
        for index, channels in enumerate(net.conv_blocks[1].core[0].keep_channels)
    }
    print_counter("Slimmable width distribution among samples reaching Conv block 2", reached_width, width_labels)
    print(f"Conv block 2 not reached: {skipped_width}")

    print_counter("MoE selection in linear block 1", expert_1, {0: "SNN", 1: "DNN"})
    reached_expert_2 = [value for value in expert_2 if value >= 0]
    skipped_expert_2 = sum(value < 0 for value in expert_2)
    print_counter("MoE selection in linear block 2 among reached samples", reached_expert_2, {0: "SNN", 1: "DNN"})
    print(f"Linear block 2 not reached: {skipped_expert_2}")

    mae = float(np.mean(np.abs(prediction - target)))
    flops = torch.cat(flops_all)
    gate_flops = torch.cat(gate_flops_all)
    summary = summarize_flops(net)
    gate_cost = gate_only_full_path_cost(net)

    print(f"\n-- Overall MAE: {mae:.4f} Hz --")
    print(f"Evaluated samples: {target.size}")
    print(
        "Actual route-dependent FLOPs per sample: "
        f"mean={flops.mean().item():.2f}, min={flops.min().item():.2f}, "
        f"max={flops.max().item():.2f}"
    )
    mean_total = flops.mean().item()
    mean_gate = gate_flops.mean().item()
    print(
        "Average executed routing overhead: "
        f"{mean_gate:.2f} FLOPs/sample; {100.0 * mean_gate / mean_total:.4f}% of the "
        f"average dynamic inference cost; {100.0 * mean_gate / (mean_total - mean_gate):.4f}% "
        "increase over the corresponding ungated computation."
    )

    route_counter = Counter(route_keys)
    most_route, most_count = route_counter.most_common(1)[0]
    representative_index = route_keys.index(most_route)
    most_total = flops[representative_index].item()
    most_gate = gate_flops[representative_index].item()
    cd, wi, e1, ld, e2 = most_route
    width_name = "not reached" if wi < 0 else f"CF{net.conv_blocks[1].core[0].keep_channels[wi]}"
    expert2_name = "not reached" if e2 < 0 else ("SNN" if e2 == 0 else "DNN")
    print(
        "Most-selected route: "
        f"CD{cd}, width={width_name}, block1={'SNN' if e1 == 0 else 'DNN'}, "
        f"LD{ld}, block2={expert2_name}; count={most_count}/{len(route_keys)} "
        f"({most_count / len(route_keys):.4f})."
    )
    print(
        "Most-selected-route overhead: "
        f"total={most_total:.2f} FLOPs, gates={most_gate:.2f} FLOPs, "
        f"gate share={100.0 * most_gate / most_total:.4f}%, "
        f"increase over ungated route={100.0 * most_gate / (most_total - most_gate):.4f}%."
    )

    print(
        "All full-path routing networks: "
        f"{gate_cost.flops:.2f} FLOPs; "
        f"{summary['gate_fraction_of_full_percent']:.4f}% of the analytical full-reference path; "
        f"{summary['ungated_to_gated_increase_percent']:.4f}% increase over its ungated counterpart."
    )


if __name__ == "__main__":
    main()
