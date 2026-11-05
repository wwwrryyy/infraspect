#include "video_ingestor.hpp"
#include <chrono>
#include <iostream>

VideoIngestor::VideoIngestor(size_t queue_capacity)
    : camera_index_(0), is_camera_(false), frame_queue_(queue_capacity) {}

VideoIngestor::~VideoIngestor() {
    stop();
}

bool VideoIngestor::openSource(const std::string& source_path) {
    source_path_ = source_path;
    is_camera_ = false;
    return true;
}

bool VideoIngestor::openCamera(int camera_index) {
    camera_index_ = camera_index;
    is_camera_ = true;
    return true;
}

void VideoIngestor::start() {
    if (is_running_) return;
    // A previous loop may have exited on its own (e.g. open failure); reap it first,
    // otherwise assigning a new std::thread to a joinable one calls std::terminate().
    if (ingestion_thread_.joinable()) {
        ingestion_thread_.join();
    }
    frame_queue_.reset();
    is_running_ = true;
    ingestion_thread_ = std::thread(&VideoIngestor::ingestionLoop, this);
}

void VideoIngestor::stop() {
    is_running_ = false;
    frame_queue_.stop();
    if (ingestion_thread_.joinable()) {
        ingestion_thread_.join();
    }
}

void VideoIngestor::ingestionLoop() {
    cv::VideoCapture cap;
    if (is_camera_) {
        cap.open(camera_index_);
    } else {
        cap.open(source_path_);
    }

    if (!cap.isOpened()) {
        std::cerr << "[VideoIngestor Error] Failed to open video source: "
                  << (is_camera_ ? std::to_string(camera_index_) : source_path_) << std::endl;
        is_running_ = false;
        return;
    }

    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_counter = 0;
    int consecutive_failures = 0;
    const int max_rewind_failures = 3;

    while (is_running_) {
        cv::Mat frame;
        bool read_success = cap.read(frame);
        if (!read_success || frame.empty()) {
            if (!is_camera_) {
                // Loop video file for continuous benchmark streaming, but give up on
                // sources that cannot yield a frame even after rewinding.
                if (++consecutive_failures >= max_rewind_failures) {
                    std::cerr << "[VideoIngestor Error] Source unreadable after rewind: "
                              << source_path_ << std::endl;
                    is_running_ = false;
                    break;
                }
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            if (++consecutive_failures >= 100) {
                std::cerr << "[VideoIngestor Error] Camera disconnected or read timed out." << std::endl;
                is_running_ = false;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        consecutive_failures = 0;

        // Deep copy: VideoCapture may reuse its internal buffer for the next read.
        frame_queue_.push(frame.clone());
        total_frames_ingested_++;
        frame_counter++;

        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(current_time - last_time).count();
        if (elapsed_sec >= 1.0) {
            current_fps_ = static_cast<double>(frame_counter) / elapsed_sec;
            frame_counter = 0;
            last_time = current_time;
        }
    }

    cap.release();
}

bool VideoIngestor::getNextFrame(cv::Mat& frame, int timeout_ms) {
    return frame_queue_.pop(frame, timeout_ms);
}
