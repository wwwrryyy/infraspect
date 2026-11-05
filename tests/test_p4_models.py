"""
test_p4_models.py
=================
Unit tests for Phase P4 model architectures:
- Stage 1 SpikingUNet & CNNUNet (reset_mechanism="none", shapes, [0, 1] bounds).
- Stage 2 SNNAutoencoder & CNNAutoencoder (negative-only baselines, shapes, bounds).
- Stage 3 SpikingMobileNetV3Classifier & CNNMobileNetV3Classifier (64x64 crops, 6 logits).
"""

import os
import torch
import pytest
import numpy as np

from models.stage1_denoiser.model import SpikingUNet, CNNUNet
from models.stage2_detector.autoencoder import SNNAutoencoder, CNNAutoencoder
from models.stage3_classifier.model import SpikingMobileNetV3Classifier, CNNMobileNetV3Classifier
from data.datasets import DEFECT_CLASSES


def test_stage1_models_shape_and_reset():
    x = torch.rand(2, 3, 128, 128)

    # SNN
    s_unet = SpikingUNet(in_channels=3, base_ch=16, num_steps=4)
    assert s_unet.out_lif.reset_mechanism == "none"

    with torch.no_grad():
        out_snn = s_unet(x)
    assert out_snn.shape == (2, 3, 128, 128)
    assert out_snn.min() >= 0.0 and out_snn.max() <= 1.0

    # CNN
    c_unet = CNNUNet(in_channels=3, base_ch=16)
    with torch.no_grad():
        out_cnn = c_unet(x)
    assert out_cnn.shape == (2, 3, 128, 128)
    assert out_cnn.min() >= 0.0 and out_cnn.max() <= 1.0


def test_stage2_autoencoders_shape():
    x = torch.rand(2, 3, 128, 128)

    # SNN Autoencoder
    s_ae = SNNAutoencoder(in_channels=3, num_steps=4)
    with torch.no_grad():
        out_s_ae = s_ae(x)
    assert out_s_ae.shape == (2, 3, 128, 128)
    assert out_s_ae.min() >= 0.0 and out_s_ae.max() <= 1.0

    # CNN Autoencoder
    c_ae = CNNAutoencoder(in_channels=3)
    with torch.no_grad():
        out_c_ae = c_ae(x)
    assert out_c_ae.shape == (2, 3, 128, 128)
    assert out_c_ae.min() >= 0.0 and out_c_ae.max() <= 1.0


def test_stage3_classifiers_shape():
    crops = torch.rand(4, 3, 64, 64)

    # Spiking MobileNetV3
    s_mob = SpikingMobileNetV3Classifier(in_channels=3, num_classes=len(DEFECT_CLASSES), num_steps=4)
    with torch.no_grad():
        logits_snn = s_mob(crops)
    assert logits_snn.shape == (4, len(DEFECT_CLASSES))

    # CNN MobileNetV3
    c_mob = CNNMobileNetV3Classifier(in_channels=3, num_classes=len(DEFECT_CLASSES))
    with torch.no_grad():
        logits_cnn = c_mob(crops)
    assert logits_cnn.shape == (4, len(DEFECT_CLASSES))


def test_model_training_step():
    # Verify backward pass computes valid gradients without NaN
    model = CNNUNet(in_channels=3, base_ch=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_stage2_and_stage3_standalone_trainers_smoke(tmp_path):
    from types import SimpleNamespace
    from models.stage2_detector.train_snn_autoencoder import train_and_export
    from models.stage3_classifier.train import train as train_stage3

    s2_args = SimpleNamespace(
        arch="cnn", checkpoint=None, data_dir=None, output_dir=str(tmp_path),
        output=str(tmp_path / "s2.onnx"), epochs=1, batch_size=4, lr=1e-3,
        in_channels=3, num_steps=4, beta=0.85, height=32, width=32, num_synthetic=10,
    )
    ckpt, onnx_path, auroc = train_and_export(s2_args)
    assert os.path.isfile(ckpt) and os.path.isfile(onnx_path)

    s3_args = SimpleNamespace(
        arch="cnn", data_dir=None, output_dir=str(tmp_path), epochs=1, batch_size=4,
        lr=1e-3, in_channels=3, num_steps=4, beta=0.85, num_synthetic=8,
    )
    assert os.path.isfile(train_stage3(s3_args))


def test_stage3_train_handles_1d_and_2d_targets(tmp_path):
    from types import SimpleNamespace
    from models.stage3_classifier.train import train as train_stage3
    from unittest.mock import patch

    # Mock dataset to return 1D integer targets instead of 2D one-hot vectors
    with patch("models.stage3_classifier.train.Stage3DefectDataset") as mock_ds:
        class Dummy1DDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            def __getitem__(self, idx):
                return torch.rand(3, 64, 64), torch.tensor(2, dtype=torch.long)

        mock_ds.return_value = Dummy1DDataset()
        s3_args = SimpleNamespace(
            arch="cnn", data_dir=None, output_dir=str(tmp_path), epochs=1, batch_size=2,
            lr=1e-3, in_channels=3, num_steps=4, beta=0.85, num_synthetic=4,
        )
        ckpt = train_stage3(s3_args)
        assert os.path.isfile(ckpt)
