# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-04

### Added
- **Dual Edge Architectures:** Support for both Spiking Neural Networks ($T=4$ unrolled Leaky Integrate-and-Fire timesteps) and Continuous CNN ($T=1$ single-pass) backbones (`--arch [snn | cnn | compare]`).
- **3-Stage Civil Edge Pipeline:**
  - **Stage 1 (Optical Restoration):** ResUNet / SpikingUNet denoiser operating on 3-channel RGB imagery under severe atmospheric corruptions (dust, fog, vignetting, salt-and-pepper).
  - **Stage 2 (Negative-Only Anomaly Detector):** Autoencoder trained strictly on nominal uncracked concrete textures to identify fine structural deviations.
  - **Stage 3 (Defect Classifier):** MobileNetV3 / SpikingMobileNetV3 classifier evaluating defect ROIs against the CODEBRIM civil damage taxonomy.
- **High-Performance C++ Core (`src_cpp/`):** PyBind11 zero-copy `cv::Mat` bridge executing pixel-wise multi-channel anomaly calculus and Gaussian smoothing with sub-millisecond latency.
- **Tactical Inspection Logic:** Automated non-maximum suppression (NMS) bounding box clustering, GSD physical width calculation (mm), and advisory AASHTO MBEI 1130 Condition State categorization (CS1–CS4).
- **Interactive Multi-View HUD & GUI:**
  - OpenCV 2x2 multi-panel Heads-Up Display (`gui_dashboard.py --gui cv`) with real-time model toggling `[C]`, sensitivity adjustment `[+/-]`, physical crack estimation `[G]`, and freeze-frame diagnostics.
  - Native DearPyGui desktop dashboard (`--gui dpg`).
  - Headless CI execution support (`--headless --frames N`).
- **Real Civil Concrete Datasets:** Ingestion and patch extractors for 40,000 real concrete images from the METU / Bimewok benchmark, 118 human-annotated `.mat` segmentation masks from CrackForest, and high-resolution surfaces from CrackSegmentation.
- **Unsupervised Zero-Leakage Validation:** 99.5th percentile MSE anomaly threshold calibration on held-out healthy validation patches.
- **CI/CD Automation:** GitHub Actions CI workflow (`.github/workflows/ci.yml`) compiling C++ PyBind11 modules, running 34 unit tests, and verifying headless benchmarks on `ubuntu-latest`.
- **Pre-Compiled ONNX Artifacts:** Static ONNX model graphs and empirical calibration JSON profiles exported in `exports/`.
