"""Evaluate a synchronized SDyNN checkpoint.

Inference is deterministic: all routers use argmax and execute only the selected
Conv-EE/linear-EE paths, widths, and SNN/DNN experts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

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
    parser.add_argument("--cutout", type=str2bool, default=False)
    parser.add_argument("--auto", type=str2bool, default=False)
    parser.add_argument("--num_lost", type=int, default=1)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--spare_gpu", type=int, default=0)
    parser.add_argument("--shuffle_test", type=str2bool, default=False)
    parser.add_argument("--report_flops", type=str2bool, default=True)
    parser.add_argument(
        "--train_module_dir",
        type=str,
        default="/SNN/CFO/train",
        help="Directory containing cfo_scnn_train.py",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=(
            "/home/leehyunjong/Wi-Fi_Preambles/stfcfo/802.11ax_synth_changing/"
            "WiFi_20MHz_L-STF_ax_cfo_rapid_chg4_mixed_test.txt"
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


def segment_maes(prediction: np.ndarray, target: np.ndarray, segments: int = 10) -> List[float]:
    indices = np.array_split(np.arange(target.size), segments)
    return [float(np.mean(np.abs(prediction[idx] - target[idx]))) for idx in indices if idx.size > 0]


def main() -> None:
    args = build_parser().parse_args()
    device = configure_device(args.gpu, args.spare_gpu)

    train_dir = str(Path(args.train_module_dir).resolve())
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    from cfo_scnn_train_v2 import apply_cutout, estimate_inference_flops, load_checkpoint

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

    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    flop_batches: List[torch.Tensor] = []

    with torch.inference_mode():
        for inputs, target in loader:
            inputs = inputs.to(device)
            if args.cutout:
                inputs = apply_cutout(inputs, args.auto, args.num_lost)
            output = net(inputs)
            prediction_hz = output.prediction * (y_max - y_min) + y_min
            predictions.append(prediction_hz.cpu().numpy())
            labels.append(target.numpy())
            if args.report_flops:
                flop_batches.append(estimate_inference_flops(output, net))
            functional.reset_net(net)

    prediction = np.concatenate(predictions, axis=0).reshape(-1)
    target = np.concatenate(labels, axis=0).reshape(-1)
    if prediction.shape != target.shape:
        raise RuntimeError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")

    for index, value in enumerate(segment_maes(prediction, target), start=1):
        print(f"MAE segment {index}: {value:.4f} Hz")
    print(f"Average MAE: {np.mean(np.abs(prediction - target)):.4f} Hz")
    print(f"Evaluated samples: {target.size}")

    if flop_batches:
        flops = torch.cat(flop_batches)
        print(
            "Actual route-dependent FLOPs per sample: "
            f"mean={flops.mean().item():.2f}, min={flops.min().item():.2f}, "
            f"max={flops.max().item():.2f}"
        )


if __name__ == "__main__":
    main()
