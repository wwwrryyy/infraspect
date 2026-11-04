"""
export.py
=========
PyTorch → ONNX export for Stage 1 Denoiser (SNN & CNN backbones).

For SNN, statically unrolls the T=4 timesteps into the computation graph so
standard ONNX Runtime can execute it on edge CPUs/GPUs with no dynamic loops.
"""

import os
import argparse
import torch

from models.stage1_denoiser.model import SpikingUNet, CNNUNet


def export_onnx(args):
    device = torch.device("cpu")

    if args.arch.lower() == "snn":
        model = SpikingUNet(
            in_channels=args.in_channels,
            base_ch=args.base_ch,
            num_steps=args.num_steps,
            beta=args.beta,
        ).to(device)
        default_out = "exports/sunet_denoiser_static_T4.onnx"
    else:
        model = CNNUNet(
            in_channels=args.in_channels,
            base_ch=args.base_ch,
        ).to(device)
        default_out = "exports/cnn_denoiser.onnx"

    out_path = args.output if args.output else default_out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded weights from: {args.checkpoint}")

    model.eval()
    dummy = torch.randn(1, args.in_channels, args.height, args.width, device=device)

    print(f"[*] Exporting Stage 1 {args.arch.upper()} to ONNX: {out_path} ...")
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_frame"],
        output_names=["restored_frame"],
        dynamic_axes={
            "input_frame": {0: "batch_size"},
            "restored_frame": {0: "batch_size"},
        },
        dynamo=False,
    )

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[+] Exported successfully: {out_path} ({size_kb:.1f} KB)")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Export Stage 1 Denoiser to ONNX")
    p.add_argument("--arch", type=str, default="snn", choices=["snn", "cnn"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
