
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "load_image.hpp"
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

std::vector<ov::Tensor> utils::load_images(const std::filesystem::path& input_path) {
    if (input_path.empty() || !fs::exists(input_path)) {
        throw std::runtime_error{"Path to images is empty or does not exist."};
    }
    if (fs::is_directory(input_path)) {
        std::set<fs::path> sorted_images{fs::directory_iterator(input_path), fs::directory_iterator()};
        std::vector<ov::Tensor> images;
        for (const fs::path& dir_entry : sorted_images) {
            images.push_back(utils::load_image(dir_entry));
        }
        return images;
    }
    return {utils::load_image(input_path)};
}

ov::Tensor utils::load_video(const std::filesystem::path& input_path) {
    auto rgbs = load_images(input_path);
    if (rgbs.size() == 0) {
        return {};
    }

    auto video = ov::Tensor(ov::element::u8,
                            ov::Shape{rgbs.size(), rgbs[0].get_shape()[1], rgbs[0].get_shape()[2], rgbs[0].get_shape()[3]});
    std::cout << "video.shape = " << video.get_shape() << std::endl;

    auto stride = rgbs[0].get_byte_size();
    std::cout << "stride = " << stride << std::endl;
    auto dst = reinterpret_cast<char*>(video.data());
    int b = 0;
    for (auto rgb : rgbs)
    {
        std::memcpy(dst + stride * b, rgb.data(), stride);
        b++;
    }
    return video;
}

ov::Tensor utils::load_and_sample_video(const std::filesystem::path& video_path) {
    // 1. 打开视频文件
    cv::VideoCapture cap(video_path.string());
    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Could not open video file: " + video_path.string());
    }

    // 2. 获取视频信息
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    // 目标抽帧频率：2 FPS
    constexpr double target_fps = 2.0;

    // 3. 计算抽帧间隔
    // 间隔帧数 = 视频原始帧率 / 目标抽帧率
    // 例如：如果视频是 30 FPS，目标是 2 FPS，则间隔为 30 / 2 = 15 帧
    if (video_fps < target_fps) {
        throw std::runtime_error("Video FPS is lower than the target sampling rate (2 FPS).");
    }
    int frame_interval = static_cast<int>(std::round(video_fps / target_fps));

    std::cout << "Video FPS: " << video_fps << ", Sampling every " << frame_interval << " frames." << std::endl;
    
    std::vector<cv::Mat> sampled_frames;
    cv::Mat frame;
    int current_frame_idx = 0;

    // 4. 循环抽帧
    while (cap.read(frame)) {
        // 检查是否到达抽帧点
        if (current_frame_idx % frame_interval == 0) {
            // 将帧转换为 RGB 格式 (OpenCV 默认是 BGR)
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            
            // 将帧复制到 vector 中
            sampled_frames.push_back(frame.clone()); 
        }
        current_frame_idx++;
    }

    // 检查是否有抽样帧
    if (sampled_frames.empty()) {
        throw std::runtime_error("No frames were sampled from the video.");
    }
    
    size_t num_sampled_frames = sampled_frames.size();
    size_t H = frame_height;
    size_t W = frame_width;
    constexpr size_t C = 3; // RGB 3通道

    // 5. 创建最终的 OpenVINO Tensor
    // 形状为 (N, H, W, C)
    ov::Shape tensor_shape{num_sampled_frames, H, W, C};
    ov::Tensor video_tensor(ov::element::u8, tensor_shape);

    // 获取 Tensor 内部数据的指针
    unsigned char* tensor_data_ptr = video_tensor.data<unsigned char>();
    size_t frame_size_bytes = H * W * C;

    // 6. 将所有抽样帧的数据复制到 Tensor 中
    for (size_t i = 0; i < num_sampled_frames; ++i) {
        // 计算当前帧在 Tensor 数据块中的起始偏移量
        unsigned char* current_dest_ptr = tensor_data_ptr + i * frame_size_bytes;
        
        // 确保帧数据是连续的，并复制数据
        if (sampled_frames[i].isContinuous()) {
            std::memcpy(current_dest_ptr, sampled_frames[i].data, frame_size_bytes);
        } else {
            // 如果数据不连续，按行复制 (虽然 cv::cvtColor 后通常是连续的)
            for (int r = 0; r < H; ++r) {
                std::memcpy(
                    current_dest_ptr + r * W * C,
                    sampled_frames[i].ptr<unsigned char>(r),
                    W * C
                );
            }
        }
    }

    std::cout << "Successfully sampled " << num_sampled_frames << " frames." << std::endl;
    return video_tensor;
}

#include <opencv2/opencv.hpp>
// Return video with shape: [num_frames, height, width, 3]
ov::Tensor utils::create_countdown_frames()
{
    int frames_count = 5, height = 240, width = 360;
    auto video = ov::Tensor(ov::element::u8,
                            ov::Shape{(size_t)frames_count, (size_t)height, (size_t)width, 3});

    for (int i = frames_count; i > 0; i--)
    {
        cv::Mat frame = cv::Mat::zeros(height, width, CV_8UC3);
        std::string text = std::to_string(i);

        int baseline = 0;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 3.0; // Python '3' is a double in C++ OpenCV
        int thickness = 4;

        // The C++ getTextSize returns the size as a cv::Size
        cv::Size textSize = cv::getTextSize(
            text,
            fontFace,
            fontScale,
            thickness,
            &baseline // baseline is passed by pointer
        );

        int text_width = textSize.width;
        int text_height = textSize.height;
        int text_x = (width - text_width) / 2;
        int text_y = (height + text_height) / 2;

        cv::Scalar color = cv::Scalar(255, 255, 255); // BGR: White
        cv::Point org(text_x, text_y);                // Origin point for the text

        cv::putText(
            frame,
            text,
            org,
            fontFace,
            fontScale,
            color,
            thickness,
            cv::LINE_AA // The line type constant
        );

        int idx = frames_count - i;
        std::memcpy((char*)video.data() + idx * height * width * 3, frame.data, height * width * 3);

        // cv::imshow("Centered Text Frame", frame);
        // cv::waitKey(0);
    }
    return video;
}

class SharedImageAllocator : public ov::Allocator {
public:
    unsigned char* image;
    size_t total_bytes; // 存储 image 数据的总字节数

    SharedImageAllocator(unsigned char* img, size_t bytes)
        : image(img), total_bytes(bytes) {}

    void* allocate(size_t bytes, size_t alignment) noexcept {
        if (bytes == total_bytes && image) {
            return image;
        }
        return nullptr; 
    }

    bool deallocate(void* handle, size_t bytes, size_t alignment) noexcept {
        if (handle == image && bytes == total_bytes) {
            stbi_image_free(image);
            image = nullptr;
            return true;
        }
        return false;
    }

    bool is_equal(const ov::Allocator& other) const noexcept {
        return static_cast<const void*>(this) == static_cast<const void*>(&other);
    }
};

ov::Tensor utils::load_image(const std::filesystem::path& image_path) {
    int x = 0, y = 0, channels_in_file = 0;
    constexpr int desired_channels = 3;

    unsigned char* image = stbi_load(
        image_path.string().c_str(),
        &x, &y, &channels_in_file, desired_channels);

    if (!image) {
        std::stringstream error_message;
        error_message << "Failed to load the image '" << image_path << "'";
        throw std::runtime_error{error_message.str()};
    }

    size_t total_bytes = size_t(x) * size_t(y) * size_t(desired_channels);

    ov::Allocator allocator = SharedImageAllocator(image, total_bytes);

    return ov::Tensor(
        ov::element::u8,
        ov::Shape{1, size_t(y), size_t(x), size_t(desired_channels)},
        allocator
    );
}