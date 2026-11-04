"""
export.py
=========
ONNX export for Stage 3 Defect Classifier (SNN and CNN).

Exports:
- spiking_mobilenet_classifier_static_T4.onnx
- cnn_mobilenet_classifier.onnx
"""

import os
import argparse
import torch

from models.stage3_classifier.model import SpikingMobileNetV3Classifier, CNNMobileNetV3Classifier
from data.datasets import DEFECT_CLASSES


def export_onnx(args):
    device = torch.device("cpu")

    if args.arch.lower() == "snn":
        model = SpikingMobileNetV3Classifier(
            in_channels=args.in_channels,
            num_classes=len(DEFECT_CLASSES),
            num_steps=args.num_steps,
            beta=args.beta,
        ).to(device)
        default_out = "exports/spiking_mobilenet_classifier_static_T4.onnx"
    else:
        model = CNNMobileNetV3Classifier(
            in_channels=args.in_channels,
            num_classes=len(DEFECT_CLASSES),
        ).to(device)
        default_out = "exports/cnn_mobilenet_classifier.onnx"

    out_path = args.output if args.output else default_out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded weights from: {args.checkpoint}")

    model.eval()
    dummy = torch.randn(1, args.in_channels, 64, 64, device=device)

    print(f"[*] Exporting Stage 3 {args.arch.upper()} to ONNX: {out_path} ...")
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["roi_crop"],
        output_names=["defect_logits"],
        dynamic_axes={
            "roi_crop": {0: "batch_size"},
            "defect_logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[+] Exported Stage 3 ONNX: {out_path} ({size_kb:.1f} KB)")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Export Stage 3 Classifier to ONNX")
    p.add_argument("--arch", type=str, default="snn", choices=["snn", "cnn"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.85)
    return p.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
