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

static const uint16_t ATTENTION_MASK = 0xF0E2; // -9984 by bfloat16

class Qwen {
public:
  void init(const std::vector<int> &devid, int eos_token_id, std::string model_path);
  void deinit();
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  std::vector<int> answer(std::vector<int> history_tokens);

  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()) {};
  int sample(const std::vector<float>& probs, const std::vector<int>& tokens);

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  std::vector<bm_tensor_t> inputs_embed_512, outputs_embed_512;
  std::vector<bm_tensor_t> inputs_pid, inputs_attention;
  std::vector<bm_tensor_t> next_inputid, next_pid, next_attention;
  std::vector<std::vector<bm_tensor_t>> past_key, past_value;
  std::vector<bm_tensor_t> inputs_lm;
  std::vector<bm_tensor_t> outputs_lm, outputs_logit_lm, outputs_token_lm;
  std::string name_embed;
  std::string name_embed_cache;
  std::string name_lm;
  std::vector<std::string> name_blocks;
  std::vector<std::string> name_blocks_cache;

  int EOS;
  int device_num;
  int token_length;
  int SEQLEN;
  int NUM_LAYERS;
  std::vector<int> visited_tokens;
};

void Qwen::init(const std::vector<int> &devices, int eos_token_id, std::string model_path) {
  // params
  device_num = devices.size();
  EOS = eos_token_id;

  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  for (auto d : devices) {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles.push_back(h);
  }
  bm_handle = handles[0];

  // create bmruntime
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // set NUM_LAYERS
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 2) / 2;

  // net names
  name_embed = "embedding";
  name_embed_cache = "embedding_cache";
  name_lm = "lm_head";
  for (int i = 0; i < NUM_LAYERS; i++) {
    name_blocks.emplace_back("block_" + std::to_string(i));
    name_blocks_cache.emplace_back("block_cache_" + std::to_string(i));
  }

  // net infos
  net_embed = bmrt_get_network_info(p_bmrt, name_embed.c_str());
  net_embed_cache = bmrt_get_network_info(p_bmrt, name_embed_cache.c_str());
  net_lm = bmrt_get_network_info(p_bmrt, name_lm.c_str());
  for (int i = 0; i < NUM_LAYERS; i++) {
    net_blocks.emplace_back(
        bmrt_get_network_info(p_bmrt, name_blocks[i].c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, name_blocks_cache[i].c_str()));
  }

  // set SEQLEN
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1];

  // resize
  net_blocks.resize(NUM_LAYERS);
  net_blocks_cache.resize(NUM_LAYERS);
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  visited_tokens.resize(SEQLEN);

  // net device mem
  inputs_embed_512.resize(net_embed->input_num);
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&inputs_embed_512[i], p_bmrt,
                        net_embed->input_loc_devices[i],
                        net_embed->input_dtypes[i],
                        net_embed->stages[0].input_shapes[i]);
    assert(true == ret);
  }

  outputs_embed_512.resize(net_embed->output_num);
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&outputs_embed_512[i], p_bmrt,
                        net_embed->output_loc_devices[i],
                        net_embed->output_dtypes[i],
                        net_embed->stages[0].output_shapes[i]);
    assert(true == ret);
  }

  next_inputid.resize(device_num);  
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&next_inputid[i], p_bmrt,
                        net_embed_cache->input_loc_devices[i],
                        net_embed_cache->input_dtypes[i],
                        net_embed_cache->stages[0].input_shapes[i]);
    assert(true == ret);
  }

  inputs_pid.resize(device_num);
  inputs_attention.resize(device_num);
  int in_num = net_blocks[0]->input_num / device_num;
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&inputs_pid[i], p_bmrt,
                        net_blocks[0]->input_loc_devices[1 + i * in_num],
                        net_blocks[0]->input_dtypes[1 + i * in_num],
                        net_blocks[0]->stages[0].input_shapes[1 + i * in_num]);
    assert(true == ret);

    ret = bmrt_tensor_ex(&inputs_attention[i], p_bmrt,
                        net_blocks[0]->input_loc_devices[2 + i * in_num],
                        net_blocks[0]->input_dtypes[2 + i * in_num],
                        net_blocks[0]->stages[0].input_shapes[2 + i * in_num]);
    assert(true == ret);
  }

  next_pid.resize(device_num);
  next_attention.resize(device_num);
  int in_num_cache = net_blocks_cache[0]->input_num / device_num;
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&next_pid[i], p_bmrt,
                        net_blocks_cache[0]->input_loc_devices[1 + i * in_num_cache],
                        net_blocks_cache[0]->input_dtypes[1 + i * in_num_cache],
                        net_blocks_cache[0]->stages[0].input_shapes[1 + i * in_num_cache]);
    assert(true == ret);

    ret = bmrt_tensor_ex(&next_attention[i], p_bmrt,
                        net_blocks_cache[0]->input_loc_devices[2 + i * in_num_cache],
                        net_blocks_cache[0]->input_dtypes[2 + i * in_num_cache],
                        net_blocks_cache[0]->stages[0].input_shapes[2 + i * in_num_cache]);
    assert(true == ret);
  }

  int out_num = net_blocks[0]->output_num / device_num;
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i].resize(device_num);
    past_value[i].resize(device_num);
    for (int j = 0; j < device_num; j++) {
      ret = bmrt_tensor_ex(&past_key[i][j], p_bmrt,
                          net_blocks[0]->output_loc_devices[1 + j * out_num],
                          net_blocks[0]->output_dtypes[1 + j * out_num],
                          net_blocks[0]->stages[0].output_shapes[1 + j * out_num]);
      assert(true == ret);
      ret = bmrt_tensor_ex(&past_value[i][j], p_bmrt,
                          net_blocks[0]->output_loc_devices[2 + j * out_num],
                          net_blocks[0]->output_dtypes[2 + j * out_num],
                          net_blocks[0]->stages[0].output_shapes[2 + j * out_num]);
      assert(true == ret);
    }
  }

  inputs_lm.resize(device_num);
  outputs_lm.resize(device_num);
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&inputs_lm[i], p_bmrt, i, net_lm->input_dtypes[0],
                        net_lm->stages[0].input_shapes[0]);
    assert(true == ret);
    ret = bmrt_tensor_ex(&outputs_lm[i], p_bmrt, i, net_lm->output_dtypes[0],
                        net_lm->stages[0].output_shapes[0]);
    assert(true == ret);
  }
}

void Qwen::deinit() {
  for (int i = 0; i < device_num; ++i) {
    bm_free_device(handles[i], inputs_embed_512[i].device_mem);
    bm_free_device(handles[i], outputs_embed_512[i].device_mem);
    bm_free_device(handles[i], inputs_pid[i].device_mem);
    bm_free_device(handles[i], inputs_attention[i].device_mem);
    bm_free_device(handles[i], next_inputid[i].device_mem);
    bm_free_device(handles[i], next_pid[i].device_mem);
    bm_free_device(handles[i], next_attention[i].device_mem);
    bm_free_device(handles[i], inputs_lm[i].device_mem);
    bm_free_device(handles[i], outputs_logit_lm[i].device_mem);
    bm_free_device(handles[i], outputs_token_lm[i].device_mem);
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; j++) {
      bm_free_device(handles[j], past_key[i][j].device_mem);
      bm_free_device(handles[j], past_value[i][j].device_mem);
    }
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

int Qwen::sample(const std::vector<float>& probs, const std::vector<int>& tokens) {
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int Qwen::forward_first(std::vector<int> &tokens) {
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());
  
  token_length = tokens.size();

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  std::vector<int> input_nums(device_num, 1);
  std::vector<void*> datas(device_num, (void*)visited_tokens.data());
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_embed_512.data(), datas.data(),
                          input_nums.data(), device_num);
  auto ret =
      bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(),
                            inputs_embed_512.data(), inputs_embed_512.size(),
                            outputs_embed_512.data(), outputs_embed_512.size(),
                            true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  std::vector<void*> pos_id_datas(device_num, position_id.data());
  std::vector<void*> in_attn_datas(device_num, attention_mask.data());
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_pid.data(), pos_id_datas.data(),
                          input_nums.data(), device_num);
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_attention.data(),in_attn_datas.data(),
                          input_nums.data(), device_num);
  auto embed_512 = outputs_embed_512;
  std::vector<bm_tensor_t> inputs_block;
  std::vector<bm_tensor_t> outputs_block;
  for (int i = 0; i < device_num; ++i) {
    embed_512[i].shape = net_blocks[0]->stages[0].input_shapes[0];
    inputs_block.push_back(embed_512[i]);
    inputs_block.push_back(inputs_pid[i]);
    inputs_block.push_back(inputs_attention[i]);
    outputs_block.push_back(embed_512[i]);
    outputs_block.push_back(past_key[0][i]);
    outputs_block.push_back(past_value[0][i]);
  }

  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; ++j) {
      outputs_block[1 + j * 3] = past_key[i][j];
      outputs_block[2 + j * 3] = past_value[i][j];
    }
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks[i].c_str(),
                                inputs_block.data(), inputs_block.size(),
                                outputs_block.data(), outputs_block.size(),
                                true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
  }

  // forward lmhead
  int bytes = embed_512[0].device_mem.size / SEQLEN;
  bm_memcpy_d2d_byte(bm_handle, inputs_lm[0].device_mem, 0,
                     embed_512[0].device_mem, (token_length - 1) * bytes,
                     bytes);
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm[0], 1,
                              &outputs_lm[0], 1, true, false);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm[0].device_mem);

  visited_tokens.emplace_back(token);
  return token;
}

int Qwen::forward_next() {
  token_length += 1;
  int cur_token = visited_tokens[visited_tokens.size() - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;

  // forward embedding
  std::vector<bm_tensor_t> inputs_embed;
  std::vector<void*> input_datas;
  std::vector<int> input_nums(device_num, 1);
  for (int i = 0; i < device_num; ++i) {
    inputs_embed.push_back(outputs_lm[i]); // token_id
    inputs_embed[i].shape = net_embed_cache->stages[0].input_shapes[0];
    input_datas.push_back((void*)(&cur_token));
  }
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_embed.data(), input_datas.data(),
                          input_nums.data(), device_num);
  auto ret = bmrt_launch_tensor_ex(p_bmrt, name_embed_cache.c_str(),
                                  inputs_embed.data(), inputs_embed.size(),
                                  inputs_lm.data(), inputs_lm.size(), true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  std::vector<void*> attn_datas(device_num, attention_mask.data());
  std::vector<void*> pid_datas(device_num, &position_id);
  bmrt_memcpy_s2d_parallel(p_bmrt, next_attention.data(), attn_datas.data(),
                          input_nums.data(), device_num);
  bmrt_memcpy_s2d_parallel(p_bmrt, next_pid.data(), pid_datas.data(),
                          input_nums.data(), device_num);
                          
  // WARNING: make inputs_lm device_num                   
  std::vector<bm_tensor_t> embed_1 = inputs_lm;
  for (int i = 0; i < device_num; ++i) {
    embed_1[i].shape = net_blocks_cache[0]->stages[0].input_shapes[0];
  }
  std::vector<bm_tensor_t> inputs_block;
  std::vector<bm_tensor_t> outputs_block;
  for (int i = 0; i < device_num; ++i) {
    inputs_block.push_back(embed_1[i]);
    inputs_block.push_back(next_pid[i]);
    inputs_block.push_back(next_attention[i]);
    inputs_block.push_back(past_key[0][i]);
    inputs_block.push_back(past_value[0][i]);
    outputs_block.push_back(embed_1[i]);
    outputs_block.push_back(past_key[0][i]);
    outputs_block.push_back(past_value[0][i]);
  }

  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; ++j) {
      inputs_block[3 + j * 5] = past_key[i][j];
      inputs_block[4 + j * 5] = past_value[i][j];
      int bytes = bm_mem_get_device_size(past_key[0][j].device_mem) / SEQLEN;
      int token_offset = (token_length - 1) * bytes;
      bm_set_device_mem(&outputs_block[1 + j * 3].device_mem, bytes,
          bm_mem_get_device_addr(past_key[i][j].device_mem) + token_offset);
      bm_set_device_mem(&outputs_block[2 + j * 3].device_mem, bytes,
          bm_mem_get_device_addr(past_value[i][j].device_mem) + token_offset);
    }
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks_cache[i].c_str(),
                                inputs_block.data(), inputs_block.size(),
                                outputs_block.data(), outputs_block.size(),
                                true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
  }

  // forward lmhead
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm[0], 1,
                              &outputs_lm[0], 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm[0].device_mem);

  visited_tokens.emplace_back(token);
  return token;
}


std::vector<int> Qwen::answer(std::vector<int> history_tokens) {
  int tok_num = 0;
  if (history_tokens.empty()) {
    printf("Sorry: your question is too wierd!!\n");
    history_tokens.clear();
    return {};
  }

  // make sure token not too large
  if ((int)history_tokens.size() > SEQLEN - 10) {
    history_tokens.clear();
    printf("Error: your question is too large!\n");
    return {};
  }

  std::vector<int> result_tokens;
  int token = forward_first(history_tokens);
  while (token != EOS && token_length < SEQLEN) {
    result_tokens.emplace_back(token);
    tok_num++;
    token = forward_next();
  }

  return result_tokens;
}

PYBIND11_MODULE(chat_parallel, m) {
    pybind11::class_<Qwen>(m, "Qwen")
        .def(pybind11::init<>())
        .def("init", &Qwen::init)
        .def("forward_first", &Qwen::forward_first)
        .def("forward_next", &Qwen::forward_next)
        .def("deinit", &Qwen::deinit);
}

