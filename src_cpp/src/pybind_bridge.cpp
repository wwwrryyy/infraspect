#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "video_ingestor.hpp"
#include "anomaly_calculus.hpp"
#include "sharp_fault_tolerance.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

namespace py = pybind11;

/**
 * Capsule-owned zero-copy memory bridge between C++ cv::Mat and NumPy ndarrays.
 * Wraps cv::Mat data pointer inside a py::capsule owner so NumPy views the exact C++ buffer
 * without copying or risking a dangling pointer.
 */
py::array_t<uint8_t> mat_to_numpy_zerocopy_u8(cv::Mat& mat) {
    if (mat.empty()) return py::array_t<uint8_t>();

    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    if (mat.channels() == 1) {
        shape = { static_cast<ssize_t>(mat.rows), static_cast<ssize_t>(mat.cols) };
        strides = { static_cast<ssize_t>(mat.step[0]), static_cast<ssize_t>(mat.step[1]) };
    } else {
        shape = { static_cast<ssize_t>(mat.rows), static_cast<ssize_t>(mat.cols), static_cast<ssize_t>(mat.channels()) };
        strides = { static_cast<ssize_t>(mat.step[0]), static_cast<ssize_t>(mat.step[1]), static_cast<ssize_t>(mat.elemSize1()) };
    }

    auto* mat_owner = new cv::Mat(mat); // Shared refcounted header
    py::capsule base(mat_owner, [](void* p) { delete static_cast<cv::Mat*>(p); });
    return py::array_t<uint8_t>(shape, strides, mat_owner->data, base);
}

py::array_t<float> mat_to_numpy_zerocopy_f32(cv::Mat& mat) {
    if (mat.empty()) return py::array_t<float>();

    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;

    if (mat.channels() == 1) {
        shape = { static_cast<ssize_t>(mat.rows), static_cast<ssize_t>(mat.cols) };
        strides = { static_cast<ssize_t>(mat.step[0]), static_cast<ssize_t>(mat.step[1]) };
    } else {
        shape = { static_cast<ssize_t>(mat.rows), static_cast<ssize_t>(mat.cols), static_cast<ssize_t>(mat.channels()) };
        strides = { static_cast<ssize_t>(mat.step[0]), static_cast<ssize_t>(mat.step[1]), static_cast<ssize_t>(mat.elemSize1()) };
    }

    auto* mat_owner = new cv::Mat(mat);
    py::capsule base(mat_owner, [](void* p) { delete static_cast<cv::Mat*>(p); });
    return py::array_t<float>(shape, strides, reinterpret_cast<float*>(mat_owner->data), base);
}

cv::Mat numpy_float_to_mat(py::array_t<float, py::array::c_style | py::array::forcecast> input_array) {
    py::buffer_info buf = input_array.request();
    if (buf.ndim < 2) {
        return cv::Mat();
    }
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);
    int channels = (buf.ndim == 3) ? static_cast<int>(buf.shape[2]) : 1;
    int type = (channels == 3) ? CV_32FC3 : CV_32FC1;

    // Direct contiguous buffer view during synchronous call
    return cv::Mat(rows, cols, type, buf.ptr);
}

/**
 * Unified Edge Inspection Engine exposed to Python.
 */
class BattlefieldEngine {
private:
    VideoIngestor ingestor_;
    AnomalyCalculus anomaly_calc_;
    SHARPFaultTolerance sharp_;

    cv::Mat current_raw_frame_rgb_;
    cv::Mat current_mse_heatmap_;
    cv::Mat current_visual_heatmap_;

public:
    BattlefieldEngine() : ingestor_(30), anomaly_calc_(0.06f, 0.8f) {}

    bool startStream(const std::string& source) {
        if (source == "0" || source == "camera") {
            ingestor_.openCamera(0);
        } else {
            ingestor_.openSource(source);
        }
        ingestor_.start();
        return true;
    }

    void stopStream() {
        ingestor_.stop();
    }

    /**
     * @brief Ingests next video frame and converts native BGR to RGB.
     * @return (has_frame: bool, raw_frame_rgb_zerocopy: np.ndarray)
     */
    py::tuple grabFrame() {
        cv::Mat raw_bgr;
        bool has_frame = ingestor_.getNextFrame(raw_bgr, 20);
        if (!has_frame || raw_bgr.empty()) {
            return py::make_tuple(false, py::none());
        }

        // Convert camera BGR to RGB for standard model ingestion
        if (raw_bgr.channels() == 3) {
            cv::cvtColor(raw_bgr, current_raw_frame_rgb_, cv::COLOR_BGR2RGB);
        } else {
            current_raw_frame_rgb_ = raw_bgr;
        }

        py::array_t<uint8_t> rgb_arr = mat_to_numpy_zerocopy_u8(current_raw_frame_rgb_);
        return py::make_tuple(true, rgb_arr);
    }

    /**
     * @brief Computes anomaly heatmap between Stage 1 restored frame and Stage 2 reconstructed baseline.
     * @return (mse_heatmap_zerocopy: np.ndarray, visual_heatmap_zerocopy: np.ndarray, mean_mse: float)
     */
    py::tuple computeAnomaly(py::array_t<float, py::array::c_style | py::array::forcecast> restored_input,
                             py::array_t<float, py::array::c_style | py::array::forcecast> recon_input,
                             float latency_ms = 15.0f) {
        cv::Mat restored_mat = numpy_float_to_mat(restored_input);
        cv::Mat recon_mat = numpy_float_to_mat(recon_input);

        float mean_mse = anomaly_calc_.computePixelwiseMSE(restored_mat, recon_mat, current_mse_heatmap_);
        anomaly_calc_.generateVisualHeatmap(current_mse_heatmap_, current_visual_heatmap_, true);

        // Telemetry update to SHARP engine with real measured latency
        float queue_fill = static_cast<float>(ingestor_.getQueueSize()) / static_cast<float>(ingestor_.getQueueCapacity() + 1e-6f);
        sharp_.updateTelemetry(queue_fill, latency_ms, mean_mse);

        py::array_t<float> mse_arr = mat_to_numpy_zerocopy_f32(current_mse_heatmap_);
        py::array_t<uint8_t> visual_arr = mat_to_numpy_zerocopy_u8(current_visual_heatmap_);

        return py::make_tuple(mse_arr, visual_arr, mean_mse);
    }

    /**
     * @brief Legacy backward-compatible processing loop.
     */
    py::tuple processNextFrame(py::array_t<float, py::array::c_style | py::array::forcecast> recon_input) {
        cv::Mat raw;
        bool has_frame = ingestor_.getNextFrame(raw, 20);
        if (!has_frame) {
            return py::make_tuple(false, py::none(), py::none(), py::none(), 0.0f);
        }

        if (raw.channels() == 3) {
            cv::cvtColor(raw, current_raw_frame_rgb_, cv::COLOR_BGR2RGB);
        } else {
            current_raw_frame_rgb_ = raw;
        }

        cv::Mat recon_mat;
        if (recon_input.size() > 0) {
            recon_mat = numpy_float_to_mat(recon_input);
        } else {
            recon_mat = current_raw_frame_rgb_.clone();
        }

        float mean_mse = anomaly_calc_.computePixelwiseMSE(current_raw_frame_rgb_, recon_mat, current_mse_heatmap_);
        anomaly_calc_.generateVisualHeatmap(current_mse_heatmap_, current_visual_heatmap_, true);

        float queue_fill = static_cast<float>(ingestor_.getQueueSize()) / static_cast<float>(ingestor_.getQueueCapacity() + 1e-6f);
        sharp_.updateTelemetry(queue_fill, 15.0f, mean_mse);

        py::array_t<uint8_t> raw_arr = mat_to_numpy_zerocopy_u8(current_raw_frame_rgb_);
        py::array_t<float> mse_arr = mat_to_numpy_zerocopy_f32(current_mse_heatmap_);
        py::array_t<uint8_t> visual_arr = mat_to_numpy_zerocopy_u8(current_visual_heatmap_);

        return py::make_tuple(true, raw_arr, mse_arr, visual_arr, mean_mse);
    }

    std::string getHealthStatus() const {
        return sharp_.getHealthStateString();
    }

    int getScaleFactor() const {
        return sharp_.getScaleFactor();
    }

    double getFPS() const {
        return ingestor_.getCurrentFPS();
    }

    size_t getQueueSize() const {
        return ingestor_.getQueueSize();
    }
};

PYBIND11_MODULE(battlefield_core, m) {
    m.doc() = "C++ Zero-Copy Hardware Ingestion and Anomaly Calculus Core for Infrastructure Inspection";

    py::class_<BattlefieldEngine>(m, "BattlefieldEngine")
        .def(py::init<>())
        .def("start_stream", &BattlefieldEngine::startStream, py::arg("source"))
        .def("stop_stream", &BattlefieldEngine::stopStream)
        .def("grab_frame", &BattlefieldEngine::grabFrame)
        .def("compute_anomaly", &BattlefieldEngine::computeAnomaly, py::arg("restored_input"), py::arg("recon_input"), py::arg("latency_ms") = 15.0f)
        .def("process_next_frame", &BattlefieldEngine::processNextFrame)
        .def("get_health_status", &BattlefieldEngine::getHealthStatus)
        .def("get_scale_factor", &BattlefieldEngine::getScaleFactor)
        .def("get_fps", &BattlefieldEngine::getFPS)
        .def("get_queue_size", &BattlefieldEngine::getQueueSize);

    py::class_<AnomalyCalculus>(m, "AnomalyCalculus")
        .def(py::init<float, float>(), py::arg("mse_threshold") = 0.06f, py::arg("gaussian_sigma") = 0.8f)
        .def("set_threshold", &AnomalyCalculus::setThreshold)
        .def("get_threshold", &AnomalyCalculus::getThreshold)
        .def("set_gaussian_sigma", &AnomalyCalculus::setGaussianSigma)
        .def("get_gaussian_sigma", &AnomalyCalculus::getGaussianSigma);
}
