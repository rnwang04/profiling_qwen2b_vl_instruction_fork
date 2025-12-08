
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/runtime/tensor.hpp>
#include <filesystem>

#include <fstream>
#include <string>
#include <sstream>
inline bool readFileToString(const std::string &filename, std::string &content)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        content.clear();
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    content = buffer.str();
    file.close();
    return true;
}

class CTestParam {
    public:
    CTestParam(){}
    void pasre_params(int argc, char *argv[])
    {
        auto help_fun = std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <dummy_input> <VIDEO_FILE> <device> <prompts>");

        if (argc == 1)
        {
            throw help_fun;
        }
    
        if (2 == argc && std::string(argv[1]) == std::string("-h"))
        {
            throw help_fun;
        }
    
        if (3 <= argc)
        {
            dummy_input = std::string(argv[2]) == "dummy";
        }
    
        if (4 <= argc)
        {
            video_path = argv[3];
        }
        if (5 <= argc)
        {
            device = argv[4];
        }
        if (6 <= argc)
        {
            if (!readFileToString(argv[5], prompt))
            {
                prompt = argv[5];
            }
        }
        if (7 <= argc)
        {
            if (!readFileToString(argv[6], prompt2))
            {
                prompt2 = argv[6];
            }
        }

        model_path = argv[1];
        print_param();
    }

    void print_param() {
        std::cout << "== Params:" << std::endl;
        std::cout << "    model_path = " << model_path << std::endl;
        std::cout << "    dummy_input = " << dummy_input << std::endl;
        std::cout << "    video_path = " << video_path << std::endl;
        std::cout << "    device = " << device << std::endl;
        std::cout << "    prompt = " << prompt << std::endl;
        std::cout << "    prompt2 = " << prompt2 << std::endl;
    }

    std::string video_path = "/home/arda/ruonan/genai-exp/coco.mp4";
    std::string model_path = "/mnt/disk2/models/ov-models/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B_genai";
    bool dummy_input = true;
    std::string device = "GPU";
    std::string prompt;
    std::string prompt2;
};

ov::Tensor load_bin_to_ov_tensor(
    const std::string& bin_file_path, 
    const ov::Shape& expected_shape,
    const ov::element::Type& expected_type) 
{
    // 1. 计算总元素数量和预期的字节大小
    size_t total_elements = std::accumulate(
        expected_shape.begin(), 
        expected_shape.end(), 
        (size_t)1, 
        std::multiplies<size_t>()
    );
    size_t element_size = expected_type.size();
    size_t total_bytes = total_elements * element_size;

    // 2. 打开文件并检查大小
    std::ifstream file(bin_file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + bin_file_path);
    }
    
    // 检查文件大小
    size_t file_size = file.tellg();
    if (file_size != total_bytes) {
        throw std::runtime_error("File size mismatch. Expected " + 
                                 std::to_string(total_bytes) + 
                                 " bytes, but found " + 
                                 std::to_string(file_size) + " bytes.");
    }
    
    // 将文件指针重置到文件开头
    file.seekg(0, std::ios::beg);

    // 3. 创建目标 ov::Tensor 并分配内存
    // 创建 Tensor 时，OpenVINO 会自动分配所需的内存块
    ov::Tensor loaded_tensor(expected_type, expected_shape);
    
    // 4. 读取数据到 Tensor 内部缓冲区
    
    // 获取 Tensor 内部缓冲区的指针
    void* tensor_data_ptr = loaded_tensor.data(); 

    // 将整个文件内容一次性读入 Tensor 的缓冲区
    file.read(static_cast<char*>(tensor_data_ptr), total_bytes);

    if (file.fail()) {
        throw std::runtime_error("Error reading data from file: " + bin_file_path);
    }
    
    std::cout << "Successfully loaded " << total_bytes << " bytes into ov::Tensor." << std::endl;
    return loaded_tensor;
}

/**
 * 读取txt文件内容到std::string
 * @param file_path txt文件路径
 * @param question 输出参数，用于存储文件内容
 * @throw std::runtime_error 当文件打开失败或读取错误时抛出异常
 */
void read_txt_to_string(const std::string& file_path, std::string& question) {
    // 清空输出字符串
    question.clear();

    // 打开文件（二进制模式避免换行符转换问题）
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }

    // 获取文件大小（用于优化内存分配）
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    if (file_size < 0) {
        throw std::runtime_error("无法获取文件大小: " + file_path);
    }
    file.seekg(0, std::ios::beg);

    // 预分配内存（提升效率）
    question.reserve(static_cast<size_t>(file_size));

    // 读取文件内容到字符串
    std::stringstream buffer;
    buffer << file.rdbuf();  // 高效读取全部内容
    if (!file.good() && !file.eof()) {  // 检查读取过程是否出错（未正常到达文件尾）
        throw std::runtime_error("文件读取错误: " + file_path);
    }

    question = buffer.str();
    // 将所有\r\n替换为\n
    size_t pos = 0;
    while ((pos = question.find("\r\n", pos)) != std::string::npos) {
        question.replace(pos, 2, "\n");  // 用\n替换\r\n（长度2→1）
        pos += 1;  // 跳过新替换的\n，避免重复处理
    }
}