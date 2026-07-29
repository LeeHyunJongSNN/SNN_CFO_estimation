"""Fast control-flow and gradient smoke test for the synchronized SDyNN code.

Run from the same directory as cfo_scnn_train.py:
    python smoke_test_sdynn.py

The test does not require a CFO dataset. It verifies:
  * training forward/backward and gradients to every router;
  * minimum-error SNN/DNN labels (0=SNN, 1=DNN);
  * deterministic repeated inference;
  * actual Conv-EE skipping of convolution block 2;
  * valid route-dependent FLOP accounting.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from cfo_scnn_train_v2 import (
    Net,
    binary_expert_routing_loss,
    estimate_inference_flops,
    expected_energy_loss,
    load_checkpoint,
    minimum_error_expert_labels,
    save_checkpoint,
    summarize_flops,
)


def require_finite_gradient(name: str, parameter: torch.nn.Parameter) -> None:
    if parameter.grad is None:
        raise AssertionError(f"No gradient reached {name}")
    if not torch.isfinite(parameter.grad).all():
        raise AssertionError(f"Non-finite gradient in {name}")


def main() -> None:
    torch.manual_seed(7)
    net = Net(num_steps=2)
    x = torch.randn(8, 2, 160)
    target = torch.rand(8, 1)

    net.train()
    output = net(x)
    loss = torch.mean((output.prediction - target) ** 2)
    for candidate, logits in zip(output.expert_candidate_predictions, output.expert_logits):
        if candidate is None:
            raise AssertionError("Training did not evaluate both experts")
        label = minimum_error_expert_labels(candidate, target)
        loss = loss + binary_expert_routing_loss(logits, label)
    loss = loss + 0.01 * expected_energy_loss(output, net)
    loss.backward()

    require_finite_gradient("Conv-EE gate", net.depth_gate_conv.fc.weight)
    require_finite_gradient("SC width gate", net.conv_blocks[1].core[0].gate_fc.weight)
    require_finite_gradient("MoE gate 1", net.linear_blocks[0].gate_linear.weight)
    require_finite_gradient("MoE gate 2", net.linear_blocks[1].gate_linear.weight)
    require_finite_gradient("linear-EE gate", net.depth_gate.fc.weight)

    # Force Conv-EE and linear-EE to depth 1, then verify that block 2 is not called.
    net.eval()
    with torch.no_grad():
        net.depth_gate_conv.fc.weight.zero_()
        net.depth_gate_conv.fc.bias.copy_(torch.tensor([10.0, -10.0]))
        net.depth_gate.fc.weight.zero_()
        net.depth_gate.fc.bias.copy_(torch.tensor([10.0, -10.0]))
        for block in net.linear_blocks:
            block.gate_linear.weight.zero_()
            block.gate_linear.bias.copy_(torch.tensor([10.0, -10.0]))  # SNN

    calls = {"conv2": 0}

    def count_conv2(*_args):
        calls["conv2"] += 1

    handle = net.conv_blocks[1].register_forward_hook(count_conv2)
    with torch.inference_mode():
        first = net(x)
        second = net(x)
    handle.remove()

    if calls["conv2"] != 0:
        raise AssertionError("Conv block 2 executed despite Conv-EE selecting depth 1")
    if not torch.equal(first.prediction, second.prediction):
        raise AssertionError("Repeated eval-mode inference is not deterministic")
    if not torch.all(first.conv_depth == 1):
        raise AssertionError("Forced Conv-EE depth was not respected")
    if not torch.all(first.linear_depth == 1):
        raise AssertionError("Forced linear-EE depth was not respected")

    # Force the complete route and verify finite FLOP counts.
    with torch.no_grad():
        net.depth_gate_conv.fc.bias.copy_(torch.tensor([-10.0, 10.0]))
        net.depth_gate.fc.bias.copy_(torch.tensor([-10.0, 10.0]))
        for block in net.linear_blocks:
            block.gate_linear.bias.copy_(torch.tensor([-10.0, 10.0]))  # DNN
        width_gate = net.conv_blocks[1].core[0].gate_fc
        width_gate.weight.zero_()
        width_gate.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 10.0]))

    with torch.inference_mode():
        full = net(x)
    flops = estimate_inference_flops(full, net)
    if not torch.isfinite(flops).all() or not torch.all(flops > 0):
        raise AssertionError("Invalid route-dependent FLOP count")

    # Explicitly verify routing-label semantics.
    candidate = torch.tensor([[0.1, 0.9], [0.9, 0.1]])  # [SNN,DNN]
    label = minimum_error_expert_labels(candidate, torch.zeros(2, 1))
    if label.tolist() != [0, 1]:
        raise AssertionError(f"Incorrect minimum-error labels: {label.tolist()}")

    # Checkpoint round trip.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.pt"
        save_checkpoint(str(path), net, -1.0, 1.0)
        loaded, _ = load_checkpoint(str(path), "cpu")
        loaded.eval()
        with torch.inference_mode():
            loaded_output = loaded(x)
        if loaded_output.prediction.shape != (8, 1):
            raise AssertionError("Checkpoint round trip failed")

    summary = summarize_flops(net)
    print("All synchronized SDyNN smoke tests passed.")
    print(f"Forced full-route FLOPs/sample: {flops[0].item():.2f}")
    print(f"Full-path gate-only FLOPs: {summary['full_path_gate_flops']:.2f}")
    print(f"Gate fraction of full reference: {summary['gate_fraction_of_full_percent']:.4f}%")


if __name__ == "__main__":
    main()
