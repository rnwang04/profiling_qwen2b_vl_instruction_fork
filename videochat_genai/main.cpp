// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// #include "load_image.hpp"
#include "my_utils.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>
#include <chrono>
namespace fs = std::filesystem;

bool print_subword(std::string &&subword)
{
    return !(std::cout << subword << std::flush);
}

void test_video(const CTestParam &param, ov::genai::VLMPipeline &pipe, ov::Tensor &video)
{
    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;

    std::string prompt = param.prompt;
    std::cout << "  prompt: " << prompt << std::endl;

    std::cout << "  test_video pass 'video' with one tensor " << std::endl;
    std::string sys_prompt;
	read_txt_to_string("./prompt.txt", sys_prompt);
    for (int i = 0; i < 1; i++)
    {
        pipe.start_chat(sys_prompt);
        std::cout << "  Loop: [" << i << "] ";
        auto t1 = std::chrono::high_resolution_clock::now();
        auto aa = pipe.generate(prompt,
                                ov::genai::images(std::vector<ov::Tensor>{video}),
                                ov::genai::generation_config(generation_config));
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << ", result: text =" << aa.texts[0].c_str() << ", score=" << aa.scores[0] << ", tm=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
        pipe.finish_chat();
    }
}

#include <openvino/genai/continuous_batching_pipeline.hpp>
int test_cb_add_request_vs_vlm(const CTestParam &param, ov::Tensor &video)
{
    
    std::cout << "== Start to load model: " << param.model_path << std::endl;
    ov::AnyMap cfg;
    cfg["ATTENTION_BACKEND"] = "SDPA";
    // cfg["ATTENTION_BACKEND"] = "PA";
    ov::genai::VLMPipeline ov_pipe(param.model_path, param.device, cfg);
    std::string prompt = param.prompt;

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;
    std::string sys_prompt;
	read_txt_to_string("./prompt.txt", sys_prompt);

    std::vector<std::string> res_vlm_vec;
    for (int i = 0; i < 10; i++) {
        ov_pipe.start_chat(sys_prompt);
        auto res_vlm_1 = ov_pipe.generate(prompt,
                                          ov::genai::images(std::vector<ov::Tensor>{video}),
                                          ov::genai::generation_config(generation_config));
        res_vlm_vec.push_back(res_vlm_1.texts[0]);
        ov_pipe.finish_chat();
    }

    auto scheduler_config = ov::genai::SchedulerConfig();
    auto cb_pipe = ov::genai::ContinuousBatchingPipeline(
        param.model_path,
        scheduler_config,
        param.device);
    auto tokenizer = cb_pipe.get_tokenizer();

    std::vector<std::string> res_cb_vec;
    for (int i = 0; i < 5; i++) {
        cb_pipe.start_chat(sys_prompt);
        auto handle = cb_pipe.add_request(i, prompt, std::vector<ov::Tensor>{video},
                                          generation_config);
        while (handle->get_status() != ov::genai::GenerationStatus::FINISHED)
            cb_pipe.step();
            cb_pipe.finish_chat();
    
        auto outputs = handle->read_all();
    
        auto res_cb_pipeline = tokenizer.decode(outputs[0].generated_ids);
        res_cb_vec.push_back(res_cb_pipeline);
    }

    for (int i = 0; i < 5; i++) {
        auto cmp_rslt = res_cb_vec[i].compare(res_vlm_vec[i].c_str());
        std::cout << "cmp_result [" << i << "] : " << cmp_rslt << std::endl;
        if (cmp_rslt != 0)
        {
            std::cout << "    =============================" << std::endl;
            std::cout << "    rslt_vlm = " << res_vlm_vec[i] << std::endl;
            std::cout << "    *****************************" << std::endl;
            std::cout << "    rslt_cb  = " << res_cb_vec[i] << std::endl;
            std::cout << "    =============================" << std::endl;
        }
    }

    return EXIT_SUCCESS;
}


int main(int argc, char *argv[])
{
    try
    {
        auto param = CTestParam();
        param.pasre_params(argc, argv);

        ov::AnyMap cfg;
        if (param.device == "GPU")
        {
            cfg.insert({ov::cache_dir("vlm_cache")});
            std::cout << "    cfg vlm_cache = " << "vlm_cache" << std::endl;
        }

        cfg["ATTENTION_BACKEND"] = "SDPA";
        // cfg["ATTENTION_BACKEND"] = "PA";
        std::cout << "== Init VLMPipeline" << std::endl;
        ov::genai::VLMPipeline pipe(param.model_path, param.device, cfg);
        std::cout << "== Finish VLMPipeline" << std::endl;
        ov::Tensor video;
        if (!param.dummy_input)
        {
            // video = utils::load_and_sample_video(param.video_path);
            // TODO: support real video load & sample & preprocess
            ov::Shape tensor_shape{64, 3, 224, 224};
            // load video features which have been preprocessed
            std::string input_path = "/home/arda/ruonan/profiling_qwen2b_vl_instruction_fork/videochat_genai/frame_input.bin";
            std::cout << "Load frames features from " << input_path << std::endl;
            video = load_bin_to_ov_tensor(input_path,
                                          tensor_shape,
                                          ov::element::f32);
        } else {
            ov::Shape tensor_shape{12, 3, 224, 224};
            video = ov::Tensor(ov::element::f32, tensor_shape); // video features which have been preprocessed
        }
        test_video(param, pipe, video);
        test_cb_add_request_vs_vlm(param, video);
    }
    
    catch (const std::exception &error)
    {
        std::cerr << "Catch exceptions: " << error.what() << '\n';
    }
    return EXIT_SUCCESS;
}