//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// Chip 3: LMHead worker for distributed Qwen3_VL pipeline.
// Listens for hidden states from Step1, runs lm_head + generate,
// sends token back to Step1.
//
//===----------------------------------------------------------------------===//

#include "bmruntime_interface.h"
#include "json.hpp"
#include "memory.h"
#include "net_helper.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <random>
#include <vector>

//===------------------------------------------------------------===//
// Generation Config
//===------------------------------------------------------------===//
struct GenerationConfig {
  std::vector<int> eos_token_id;
  float repetition_penalty = 1.0;
  float temperature = 1.0;
  int top_k = 50;
  float top_p = 1.0;
  std::vector<std::string> stop_strings;
  static GenerationConfig from_json(const std::string &path) {
    GenerationConfig config;
    std::ifstream in(path);
    if (!in.is_open())
      return config;
    nlohmann::json j;
    in >> j;
    if (j.contains("eos_token_id"))
      config.eos_token_id = j["eos_token_id"].get<std::vector<int>>();
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
// Step2 LMHead Runner
//===------------------------------------------------------------===//
class Step2LMHead {
public:
  void init(int devid, const std::string &model_path,
            const std::string &config_path, bool do_sample);
  void deinit();
  int forward_lmhead(const std::vector<uint16_t> &hidden_state);
  void reset_visited_tokens();
  void set_visited_tokens(const std::vector<int> &tokens);
  void add_visited_token(int token);

  int HIDDEN_SIZE;
  bool lmhead_with_topk;
  bool do_sample_flag = false;
  bm_handle_t bm_handle;

private:
  void *p_bmrt;
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_greedy_head = nullptr;
  const bm_net_info_t *net_sample_head = nullptr;
  bm_device_mem_t dev_buffer;

  // Sampling state
  std::mt19937 sgen;
  std::vector<int> visited_tokens;
  int token_length = 0;
  float penalty = 1.0;
  float temperature = 1.0;
  int top_k = 50;
  float top_p = 1.0;

  void init_tensors(const bm_net_info_t *net,
                    std::vector<bm_tensor_t> &in_tensors,
                    std::vector<bm_tensor_t> &out_tensors, int stage = 0);
  void net_launch(const bm_net_info_t *net,
                  const std::vector<bm_tensor_t> &in_tensors,
                  std::vector<bm_tensor_t> &out_tensors);
  int generate(bm_device_mem_t &logits_mem);
  int greedy_search(bm_device_mem_t &logits_mem);
  int penalty_sample(bm_device_mem_t &logits_mem);
};

void Step2LMHead::init_tensors(const bm_net_info_t *net,
                               std::vector<bm_tensor_t> &in_tensors,
                               std::vector<bm_tensor_t> &out_tensors,
                               int stage) {
  in_tensors.resize(net->input_num);
  out_tensors.resize(net->output_num);
  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[stage].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[stage].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net->stages[stage].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[stage].output_shapes[i]);
  }
}

void Step2LMHead::net_launch(const bm_net_info_t *net,
                             const std::vector<bm_tensor_t> &in_tensors,
                             std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
}

void Step2LMHead::init(int devid, const std::string &model_path,
                       const std::string &config_path, bool do_sample) {
  sgen = std::mt19937(std::random_device()());
  printf("[Step2] Device [%d] loading ...\n", devid);
  bm_status_t status = bm_dev_request(&bm_handle, devid);
  assert(BM_SUCCESS == status);

  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  printf("[Step2] Model [%s] loading ...\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  bm_thread_sync(bm_handle);
  printf("[Step2] Model loaded.\n");

  // Discover networks
  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0)
        return true;
    }
    return false;
  };
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);

  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  assert(net_lm && "lm_head network not found");

  if (is_exist("greedy_head", net_names, num_nets)) {
    net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  }
  if (is_exist("sample_head", net_names, num_nets)) {
    net_sample_head = bmrt_get_network_info(p_bmrt, "sample_head");
  }
  free(net_names);

  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;
  printf("[Step2] HIDDEN_SIZE=%d, lmhead_with_topk=%d\n", HIDDEN_SIZE,
         lmhead_with_topk);

  // Allocate device buffer
  auto buffer_size = bm_mem_get_device_size(net_lm->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);

  // Sampling config
  do_sample_flag = do_sample;
  if (do_sample) {
    if (!net_sample_head) {
      printf("[Step2] Warning: sample_head not found, using greedy.\n");
      do_sample_flag = false;
    } else if (!config_path.empty()) {
      std::string gen_path = config_path + "/generation_config.json";
      printf("[Step2] Loading generation config: %s\n", gen_path.c_str());
      auto gen_config = GenerationConfig::from_json(gen_path);
      penalty = gen_config.repetition_penalty;
      temperature = gen_config.temperature;
      top_k = gen_config.top_k;
      top_p = gen_config.top_p;
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[2],
                    (void *)&penalty);
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[3],
                    (void *)&temperature);
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[4],
                    (void *)&top_k);
      bm_memcpy_s2d(bm_handle, net_sample_head->stages[0].input_mems[5],
                    (void *)&top_p);
      printf(
          "[Step2] Sampling: penalty=%.2f, temp=%.2f, top_k=%d, top_p=%.2f\n",
          penalty, temperature, top_k, top_p);
    }
  }

  // Allocate visited_tokens buffer (max SEQLEN, use large default)
  visited_tokens.resize(8192, 0);
  token_length = 0;
}

void Step2LMHead::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void Step2LMHead::reset_visited_tokens() {
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  token_length = 0;
}

void Step2LMHead::set_visited_tokens(const std::vector<int> &tokens) {
  if ((int)tokens.size() > (int)visited_tokens.size()) {
    visited_tokens.resize(tokens.size() * 2, 0);
  }
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.begin());
  token_length = tokens.size();
}

void Step2LMHead::add_visited_token(int token) {
  if (token_length >= (int)visited_tokens.size()) {
    visited_tokens.resize(visited_tokens.size() * 2, 0);
  }
  visited_tokens[token_length++] = token;
}

int Step2LMHead::generate(bm_device_mem_t &logits_mem) {
  int token = 0;
  if (lmhead_with_topk) {
    bm_memcpy_d2s_partial(bm_handle, (void *)&token, logits_mem, sizeof(int));
  } else if (do_sample_flag && net_sample_head) {
    token = penalty_sample(logits_mem);
  } else if (net_greedy_head) {
    token = greedy_search(logits_mem);
  } else {
    // Fallback: read first value
    bm_memcpy_d2s_partial(bm_handle, (void *)&token, logits_mem, sizeof(int));
  }
  return token;
}

int Step2LMHead::greedy_search(bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_greedy_head, in_tensors, out_tensors);
  in_tensors[0].device_mem = logits_mem;
  net_launch(net_greedy_head, in_tensors, out_tensors);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_tensors[0].device_mem);
  return token;
}

int Step2LMHead::penalty_sample(bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_sample_head, in_tensors, out_tensors);
  in_tensors[0].device_mem = logits_mem;

  bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                        (void *)visited_tokens.data(),
                        token_length * sizeof(int));
  in_tensors[1].shape.dims[1] = token_length;

  net_launch(net_sample_head, in_tensors, out_tensors);

  int candidate_num = top_k;
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s_partial_offset(bm_handle, probs.data(),
                               out_tensors[0].device_mem, top_k * sizeof(float),
                               0);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s_partial_offset(bm_handle, tokens.data(),
                               out_tensors[1].device_mem, top_k * sizeof(int),
                               0);

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int Step2LMHead::forward_lmhead(const std::vector<uint16_t> &hidden_state) {
  // Upload hidden state to device
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_lm, in_tensors, out_tensors);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)hidden_state.data(),
                        HIDDEN_SIZE * sizeof(uint16_t));
  out_tensors[0].device_mem = dev_buffer;
  net_launch(net_lm, in_tensors, out_tensors);

  int token = generate(dev_buffer);
  add_visited_token(token);
  return token;
}

//===------------------------------------------------------------===//
// Main event loop
//===------------------------------------------------------------===//
void Usage() {
  printf("Step2 (LMHead Worker) Usage:\n"
         "  -m, --model     : LMHead bmodel path\n"
         "  -c, --config    : Config path (for generation_config.json)\n"
         "  -d, --devid     : Device ID (default 0)\n"
         "  -p, --port      : Listen port for Step1 (default 10002)\n"
         "  -s, --do_sample : Enable sampling\n"
         "  -h, --help      : Show help\n");
}

int main(int argc, char *argv[]) {
  std::string model_path;
  std::string config_path;
  int devid = 0;
  int listen_port = 10002;
  bool do_sample = false;

  struct option long_opts[] = {{"model", required_argument, nullptr, 'm'},
                               {"config", required_argument, nullptr, 'c'},
                               {"devid", required_argument, nullptr, 'd'},
                               {"port", required_argument, nullptr, 'p'},
                               {"do_sample", no_argument, nullptr, 's'},
                               {"help", no_argument, nullptr, 'h'},
                               {nullptr, 0, nullptr, 0}};
  int opt;
  while ((opt = getopt_long(argc, argv, "m:c:d:p:sh", long_opts, nullptr)) !=
         -1) {
    switch (opt) {
    case 'm':
      model_path = optarg;
      break;
    case 'c':
      config_path = optarg;
      break;
    case 'd':
      devid = atoi(optarg);
      break;
    case 'p':
      listen_port = atoi(optarg);
      break;
    case 's':
      do_sample = true;
      break;
    case 'h':
      Usage();
      return 0;
    default:
      Usage();
      return 1;
    }
  }
  if (model_path.empty()) {
    Usage();
    return 1;
  }

  // Initialize model
  Step2LMHead lmhead;
  lmhead.init(devid, model_path, config_path, do_sample);

  // Start listening for Step1
  int server_fd = create_server(listen_port);
  assert(server_fd >= 0);
  printf("[Step2] Waiting for Step1 connection on port %d ...\n", listen_port);
  int step1_fd = accept_client(server_fd);
  assert(step1_fd >= 0);
  printf("[Step2] Step1 connected.\n");

  // Main loop
  while (true) {
    // First read msg_type only (4 bytes)
    int32_t msg_type = 0;
    if (!recv_all(step1_fd, &msg_type, sizeof(msg_type))) {
      printf("[Step2] Step1 disconnected.\n");
      break;
    }

    if (msg_type == MSG_SHUTDOWN) {
      printf("[Step2] Shutdown received.\n");
      break;
    }

    if (msg_type == MSG_CLEAR) {
      printf("[Step2] Clear visited tokens.\n");
      lmhead.reset_visited_tokens();
      continue;
    }

    // Read remaining LMHeadMeta fields (hidden_size + visited_token_count)
    LMHeadMeta meta;
    meta.msg_type = msg_type;
    recv_all(step1_fd, &meta.hidden_size, sizeof(LMHeadMeta) - sizeof(int32_t));

    if (msg_type == MSG_PREFILL) {
      // Receive visited_tokens if present
      if (meta.visited_token_count > 0) {
        std::vector<int> vtokens(meta.visited_token_count);
        recv_all(step1_fd, vtokens.data(),
                 meta.visited_token_count * sizeof(int));
        lmhead.set_visited_tokens(vtokens);
      } else {
        lmhead.reset_visited_tokens();
      }
    }

    // For both PREFILL and DECODE: receive hidden_state
    std::vector<uint16_t> hidden_state(meta.hidden_size);
    recv_all(step1_fd, hidden_state.data(),
             meta.hidden_size * sizeof(uint16_t));

    // Run lm_head
    int token = lmhead.forward_lmhead(hidden_state);

    // Send token back to Step1
    TokenMsg tmsg;
    tmsg.token = token;
    send_all(step1_fd, &tmsg, sizeof(tmsg));
  }

  lmhead.deinit();
  close(step1_fd);
  close(server_fd);
  printf("[Step2] Exited.\n");
  return 0;
}
