//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "lmhead.hpp"
#include "json.hpp"
#include <fstream>
#include <iostream>

static void print_devmem_info(bm_handle_t &bm_handle) {
  bm_dev_stat_t stat;
  auto ret = bm_get_stat(bm_handle, &stat);
  if (ret != BM_SUCCESS) {
    std::cerr << "Failed to get device status" << std::endl;
    return;
  }
  std::cout << "DevMem: " << stat.mem_used << "/" << stat.mem_total << " MB"
            << std::endl;
}

//===------------------------------------------------------------===//
// Generation Config
//===------------------------------------------------------------===//
struct GenerationConfig {
  float repetition_penalty = 1.0;
  float temperature = 1.0;
  int top_k = 50;
  float top_p = 1.0;
  std::vector<std::string> stop_strings;
  static GenerationConfig from_json(const std::string &path) {
    GenerationConfig config;
    std::ifstream in(path);
    nlohmann::json j;
    in >> j;
    if (j.contains("repetition_penalty"))
      config.repetition_penalty = j["repetition_penalty"].get<float>();
    if (j.contains("temperature"))
      config.temperature = j["temperature"].get<float>();
    if (j.contains("top_k"))
      config.top_k = j["top_k"].get<int>();
    if (j.contains("top_p"))
      config.top_p = j["top_p"].get<float>();
    if (j.contains("stop_strings"))
      config.stop_strings = j["stop_strings"].get<std::vector<std::string>>();
    return config;
  }
};

//===------------------------------------------------------------===//
// LmHead
//===------------------------------------------------------------===//

static bool ends_with(const std::string &str, const std::string &suffix) {
  if (str.size() < suffix.size())
    return false;
  return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

bool LmHead::check_stop(const std::string &text) {
  for (const auto &stop_str : stop_strings) {
    if (ends_with(text, stop_str)) {
      return true;
    }
  }
  return false;
}

void LmHead::init_by_names() {
  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0) {
        return true;
      }
    }
    return false;
  };

  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);

  net_greedy_head = nullptr;
  if (is_exist("greedy_head", net_names, num_nets)) {
    net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  }
  net_sample_head = nullptr;
  if (is_exist("sample_head", net_names, num_nets)) {
    net_sample_head = bmrt_get_network_info(p_bmrt, "sample_head");
  }
  free(net_names);

  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;
}

void LmHead::init(int dev_id, std::string model_path, std::string config_path,
                  bool do_sample_) {
  // request bm_handle
  std::cout << "Device [ " << dev_id << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  assert(BM_SUCCESS == status);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  bm_thread_sync(bm_handle);
  printf("Done!\n");
  print_devmem_info(bm_handle);

  init_by_names();

  auto buffer_size = bm_mem_get_device_size(net_lm->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);

  do_sample = do_sample_;
  if (do_sample) {
    if (!net_sample_head) {
      std::cerr
          << "\nWarning: Sample head not found in the model. You need compile "
             "bmodel with --do_sample. Using greedy mode instead!\n";
    } else {
      std::string generation_path = config_path + "/generation_config.json";
      std::cout << "Generation Config [" << generation_path.c_str()
                << "] loading .... ";
      auto gen_config = GenerationConfig::from_json(generation_path);
      penalty = gen_config.repetition_penalty;
      temperature = gen_config.temperature;
      top_k = gen_config.top_k;
      top_p = gen_config.top_p;
      if (!gen_config.stop_strings.empty()) {
        stop_strings = gen_config.stop_strings;
      }
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[2],
                    (void *)&penalty);
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[3],
                    (void *)&temperature);
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[4],
                    (void *)&top_k);
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[5],
                    (void *)&top_p);
      std::cout << "Done!" << std::endl;
    }
  }
}

void LmHead::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

int LmHead::greedy_search(bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_greedy_head, in_tensors, out_tensors);
  in_tensors[0].device_mem = logits_mem;
  net_launch(p_bmrt, net_greedy_head, in_tensors, out_tensors);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_tensors[0].device_mem);
  return token;
}

int LmHead::penalty_sample(bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_sample_head, in_tensors, out_tensors);
  in_tensors[0].device_mem = logits_mem;

  // repeat_penalty + top_p + top_k + temperature
  bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                        (void *)visited_tokens.data(),
                        token_length * sizeof(int));
  in_tensors[1].shape.dims[1] = token_length;

  // inference
  net_launch(p_bmrt, net_sample_head, in_tensors, out_tensors);

  // get logit & token
  int candidate_num = top_k;
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s_partial_offset(bm_handle, probs.data(),
                               out_tensors[0].device_mem, top_k * sizeof(float),
                               0);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s_partial_offset(bm_handle, tokens.data(),
                               out_tensors[1].device_mem, top_k * sizeof(float),
                               0);

  // sample
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int LmHead::generate(bm_device_mem_t &logits_mem) {
  int token = 0;
  if (lmhead_with_topk) {
    bm_memcpy_d2s_partial(bm_handle, (void *)&token, logits_mem, sizeof(int));
  } else if (do_sample && net_sample_head) {
    token = penalty_sample(logits_mem);
  } else {
    token = greedy_search(logits_mem);
  }
  return token;
}

int LmHead::forward(ArrayUint16 &hidden_states) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_lm, in_tensors, out_tensors);
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)hidden_states.data(), bytes);
  out_tensors[0].device_mem = dev_buffer;
  net_launch(p_bmrt, net_lm, in_tensors, out_tensors);
  int token = generate(dev_buffer);
  visited_tokens[token_length] = token;
  token_length++;
  return token;
}
