"""
test_p1_cpp.py
==============
Unit tests for Phase P1 C++ engine and pybind11 zero-copy bridge:
- Zero-copy capsule owner verification.
- C++ AnomalyCalculus vs NumPy reference parity.
- Stream error handling and non-crash resilience.
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, "src_python")
import battlefield_core
from src_python.tactical_logic import compute_pixelwise_mse_numpy


def test_cpp_numpy_anomaly_parity():
    rng = np.random.default_rng(42)
    # 3-channel RGB float32 in [0, 1]
    restored = rng.uniform(0.1, 0.9, (64, 64, 3)).astype(np.float32)
    recon = rng.uniform(0.1, 0.9, (64, 64, 3)).astype(np.float32)

    engine = battlefield_core.BattlefieldEngine()
    mse_cpp, vis_cpp, mean_cpp = engine.compute_anomaly(restored, recon, 12.0)

    mean_py, mse_py = compute_pixelwise_mse_numpy(restored, recon, gaussian_sigma=0.8)

    # Parity check between C++ AnomalyCalculus and NumPy reference
    assert np.isclose(mean_cpp, mean_py, atol=1e-3)
    assert np.allclose(mse_cpp, mse_py, atol=5e-3)
    assert vis_cpp.shape == (64, 64, 3) # Jet colormap output


def test_zero_copy_capsule_base():
    engine = battlefield_core.BattlefieldEngine()
    a = np.zeros((32, 32, 3), dtype=np.float32)
    b = np.ones((32, 32, 3), dtype=np.float32)

    mse_arr, vis_arr, _ = engine.compute_anomaly(a, b, 10.0)

    # Zero-copy verification: array must have a valid base owner object (py::capsule)
    assert mse_arr.base is not None
    assert vis_arr.base is not None


def test_ingestor_invalid_source_non_crash():
    engine = battlefield_core.BattlefieldEngine()
    # Invalid non-existent file must not crash process or call std::terminate
    engine.start_stream("/non/existent/path/to/video.mp4")
    has_frame, frame = engine.grab_frame()
    assert not has_frame
    assert frame is None
    engine.stop_stream()
