//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "memory.h"
#include "bmruntime_interface.h"
#include <getopt.h>
#include <stdio.h>
#include <inttypes.h>
#include <random>
#include <numeric>
#include "utils.h"
#include <limits>
#include <cstring>

// static const uint16_t ATTENTION_MASK = 0xC61C; // -9984 by bfloat16
static const float ATTENTION_MASK = -10000.;

class Qwen2VL {
public:
  void init(int devid, std::string model_path);
  void deinit();
  int forward_first(std::vector<int> &tokens, std::vector<int> &position_id,
                    std::vector<float> &pixel_values, std::vector<int> &posids,
                    std::vector<float> &attnmask, int img_offset, int pixel_num,
                    std::vector<float> &vit_out_ref);
  int forward_next();

  std::mt19937 sgen;
  Qwen2VL() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
                          std::vector<int> &input_tokens, int &token_length);
  uint16_t mask_value;

public:
  int token_length;
  int SEQLEN; // read from bmodel
  int HIDDEN_SIZE;
  int NUM_LAYERS; // read from bmodel
  uint64_t IMAGE_BYTES;
  std::vector<std::vector<int>> POSITION_IDS;
  int MAX_POS = 0;
  std::string generation_mode;
  int MAX_PIXELS;
  uint64_t VIT_DIMS;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_penalty_sample_head;
  const bm_net_info_t *net_vit;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void Qwen2VL::net_launch(const bm_net_info_t *net, int stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
  }
  
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void Qwen2VL::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Qwen2VL::init(int dev_id, std::string model_path) {

  // request bm_handle
  std::cout << "Device [ " << dev_id << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  assert(BM_SUCCESS == status);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  net_penalty_sample_head = bmrt_get_network_info(p_bmrt, "penalty_sample_head");
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 6) / 2;
  IMAGE_BYTES = bm_mem_get_device_size(net_vit->stages[0].input_mems[0]);
  MAX_PIXELS = net_vit->stages[0].input_shapes[0].dims[0];

  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  auto buffer_size = bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);

  mask_value = fp32_to_uint16(ATTENTION_MASK, net_blocks[0]->input_dtypes[0]);
}

void Qwen2VL::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void Qwen2VL::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device(
      &in_tensors[0], logits_mem,
      net->input_dtypes[0], net->stages[0].input_shapes[0]);

  for (int i = 1; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[0].input_mems[i],
        net->input_dtypes[i], net->stages[0].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[0].output_mems[i],
        net->output_dtypes[i], net->stages[0].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

int Qwen2VL::greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

// int Qwen2VL::penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
//                           std::vector<int> &input_tokens, int &token_length) {
//   auto &in1_mem = net->stages[0].input_mems[1];
//   auto &in2_mem = net->stages[0].input_mems[2];
//   auto &in3_mem = net->stages[0].input_mems[3];
//   auto &in4_mem = net->stages[0].input_mems[4];
//   auto &out0_mem = net->stages[0].output_mems[0];
//   auto &out1_mem = net->stages[0].output_mems[1];

//   // repeat_penalty + top_p + top_k + temperature
//   std::vector<int> generated_tokens(SEQLEN, input_tokens[token_length - 1]);
//   repeat_last_n = std::min(repeat_last_n, token_length);
//   std::copy(input_tokens.begin() + token_length - repeat_last_n,
//             input_tokens.begin() + token_length, generated_tokens.begin());
//   bm_memcpy_s2d(bm_handle, in1_mem, (void *)generated_tokens.data());
//   bm_memcpy_s2d(bm_handle, in2_mem, (void *)&top_p);
//   bm_memcpy_s2d(bm_handle, in3_mem, (void *)&temperature);
//   bm_memcpy_s2d(bm_handle, in4_mem, (void *)&repeat_penalty);

//   // inference
//   head_launch(net, logits_mem, 0);

//   // get logit & token
//   int candidate_num = net->stages[0].output_shapes[0].dims[1];
//   std::vector<float> probs(candidate_num);
//   bm_memcpy_d2s(bm_handle, probs.data(), out0_mem);
//   std::vector<int> tokens(candidate_num);
//   bm_memcpy_d2s(bm_handle, tokens.data(), out1_mem);

//   // penalty_sample
//   std::discrete_distribution<> dist(probs.begin(), probs.end());
//   return tokens[dist(sgen)];
// }

int Qwen2VL::forward_first(std::vector<int> &tokens, std::vector<int> &position_ids,
                             std::vector<float> &pixel_values, std::vector<int> &posids,
                             std::vector<float> &attnmask, int img_offset, int pixel_num,
                             std::vector<float> &vit_out_ref) {
  std::vector<int> input_ids(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, 0);
  std::copy(tokens.begin(), tokens.end(), input_ids.data());
  POSITION_IDS.resize(position_ids.size()/3, std::vector<int>(3));
  MAX_POS = 0;
  std::vector<int> p_ids(SEQLEN*3, 0);

  std::vector<uint16_t> input_vit(500*3584, 0);
  for (size_t i = 0; i < vit_out_ref.size(); ++i)
    input_vit[i] = fp32_to_bf16_bits(vit_out_ref[i]);

  std::vector<float> pixel_values_pad(MAX_PIXELS * VIT_DIMS, 0);
  std::copy(pixel_values.begin(), pixel_values.end(), pixel_values_pad.data());

  for (int i = 0; i < (int)POSITION_IDS.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      if (MAX_POS < position_ids[j * (int)POSITION_IDS.size() + i])
        MAX_POS = position_ids[j * (int)POSITION_IDS.size() + i];
      POSITION_IDS[i][j] = position_ids[j * (int)POSITION_IDS.size() + i];
    }
  }

  token_length = tokens.size(); // text input length

  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      } else {
        attention_mask[i * SEQLEN + j] = mask_value;
      }
    }
  }

  std::vector<float> vit_attention_mask(MAX_PIXELS * MAX_PIXELS, 0);
  for (int i = 0; i < MAX_PIXELS * MAX_PIXELS; i++) {
    if (attnmask[i] != 0) {
      vit_attention_mask[i] = std::numeric_limits<float>::lowest();
    }
  }

  // forward embeding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)input_ids.data());
  net_launch(net_embed); // prefil embedding

  auto start = std::chrono::high_resolution_clock::now();
  if (img_offset > 0) {
    d2d(dev_buffer, out_mem);
    out_mem = dev_buffer;
    // forward vision transformer
    auto &vit_in_mem_pixels = net_vit->stages[0].input_mems[0];
    auto &vit_in_mem_posids = net_vit->stages[0].input_mems[1];
    auto &vit_in_mem_attnmask = net_vit->stages[0].input_mems[2];
    auto &vit_out_mem = net_vit->stages[0].output_mems[0];
    bm_memcpy_s2d(bm_handle, vit_in_mem_pixels, (void *)pixel_values_pad.data());
    bm_memcpy_s2d(bm_handle, vit_in_mem_posids, (void *)posids.data());
    bm_memcpy_s2d(bm_handle, vit_in_mem_attnmask, (void *)vit_attention_mask.data());
    // dump_net_input_to_file(bm_handle, net_vit, "vit_input.npz");
    net_launch(net_vit);

    // concatenante texting embedding and image embedding
    int dst_offset = img_offset * HIDDEN_SIZE * 2;
    int vit_size = pixel_num * HIDDEN_SIZE * sizeof(uint16_t);
    // bm_memcpy_d2d_byte(bm_handle, out_mem, dst_offset, vit_out_mem, 0, vit_size);

    int cnt = bm_mem_get_device_size(vit_out_mem) / 4;
    auto buffer = std::make_unique<float[]>(cnt);
    bm_memcpy_d2s(bm_handle, buffer.get(), vit_out_mem);
    std::vector<uint16_t> uint16_value(cnt, 0);
    for (int i = 0; i < cnt; ++i)
      uint16_value[i] = fp32_to_bf16_bits(buffer[i]);
    auto buffer_size = bm_mem_get_device_size(vit_out_mem);
    bm_device_mem_t bf16_buffer;
    bm_status_t status = bm_malloc_device_byte(bm_handle, &bf16_buffer, buffer_size/2);
    assert(BM_SUCCESS == status);
    bm_memcpy_s2d(bm_handle, bf16_buffer, (void *)uint16_value.data());
    // compare_similarity(bm_handle, bf16_buffer, net_blocks[0]->input_dtypes[0], "/workspace/LLM-TPU/models/Qwen2_VL/python_demo_video/compile/vit_out_mem.npz", "tensor");

    bm_memcpy_d2d_byte(bm_handle, out_mem, dst_offset, bf16_buffer, 0,
                       pixel_num * HIDDEN_SIZE * 2);

    // bm_memcpy_s2d(bm_handle, bf16_buffer, (void *)input_vit.data());
    // bm_memcpy_d2d_byte(bm_handle, out_mem, dst_offset, bf16_buffer, 0,
    //                    pixel_num * HIDDEN_SIZE * 2);
    // compare_similarity(bm_handle, bf16_buffer, net_blocks[0]->input_dtypes[0], "/workspace/LLM-TPU/models/Qwen2_VL/python_demo_video/compile/vit_out_mem.npz", "tensor");
    // compare_similarity(bm_handle, out_mem, net_blocks[0]->input_dtypes[0], "/workspace/LLM-TPU/models/Qwen2_VL/python_demo_video/compile/llm_input.npz", "tensor");
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "vit_launch execution time: " << duration.count() << " seconds" << std::endl;

  int out_mem_num = bm_mem_get_device_size(out_mem) / 2;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    // d2d(in0_mem, block_out_mem);
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_ids.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    // std::string ref_file_name = "/workspace/LLM-TPU/models/Qwen2_VL/python_demo_video/compile/block_";
    // ref_file_name += std::to_string(idx);
    // ref_file_name += ".npz";
    // compare_similarity(bm_handle, out_mem, net_blocks[0]->input_dtypes[0], ref_file_name, "tensor");
    
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
  }
  // compare_similarity(bm_handle, out_mem, net_blocks[0]->input_dtypes[0], "/workspace/LLM-TPU/models/Qwen2_VL/python_demo_video/compile/ref.npz", "tensor");

  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * bytes, bytes);
  net_launch(net_lm);
  int token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else {
    // token = penalty_sample(net_penalty_sample_head, lm_out_mem, total_tokens,
    //                        token_length);
    throw std::runtime_error("ERROR: unsupported generation mode!\n");
  }
  token_length++;
  return token;
}

int Qwen2VL::forward_next() {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  MAX_POS++;
  std::vector<int> token_pos = {MAX_POS, MAX_POS, MAX_POS};

  // embedding
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];

  auto &greedy_out_mem = net_greedy_head->stages[0].output_mems[0];
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  d2d(in_mem, greedy_out_mem);
  net_launch(net_embed_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (token_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)token_pos.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    } else {
      d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
      d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
    }

    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       bytes);
  }

  // forward lmhead
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm);

  int token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else {
    std::cout << "ERROR: unsupported generation mode!" << std::endl;
    exit(1);
  }
  // bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  token_length++;
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen2VL>(m, "Qwen2VL")
      .def(pybind11::init<>())
      .def("init", &Qwen2VL::init)
      .def("forward_first", &Qwen2VL::forward_first)
      .def("forward_next", &Qwen2VL::forward_next)
      .def("deinit", &Qwen2VL::deinit)
      .def_readwrite("SEQLEN", &Qwen2VL::SEQLEN) // read SEQLEN in pipeline.py
      .def_readwrite("token_length", &Qwen2VL::token_length)
      .def_readwrite("generation_mode", &Qwen2VL::generation_mode);
}