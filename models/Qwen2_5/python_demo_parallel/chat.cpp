//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "bmruntime_interface.h"
#include "memory.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

static const uint16_t BF16_NEG_10000 = 0xC61C; // -9984 by bfloat16

class Qwen {
public:
  void init(const std::vector<int> &devices, const std::string &model_path);
  void chat();
  void deinit();
  int forward_first(std::vector<int> &tokens);
  int forward_next(int cur_token);

private:
  void answer(const std::string &input_str);
  void net_launch(const std::string &net_name, std::vector<bm_tensor_t> &inputs,
                  std::vector<bm_tensor_t> &outputs, int stage_idx = 0);

public:
  int device_num;
  int TOKEN_LEN;
  int token_length;
  int SEQLEN;     // read from bmodel
  int NUM_LAYERS; // read from bmodel
  bool is_dynamic;
  std::vector<std::string> history;

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::string name_embed;
  std::string name_embed_cache;
  std::string name_lm;
  std::vector<std::string> name_blocks;
  std::vector<std::string> name_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;

  std::vector<bm_tensor_t> inputs_embed, inputs_embed_cache;
  std::vector<bm_tensor_t> hidden_states, hidden_states_cache;
  std::vector<bm_tensor_t> inputs_pid, next_pid;
  std::vector<bm_tensor_t> inputs_attention, next_attention;
  std::vector<std::vector<bm_tensor_t>> past_keys, past_values;
  std::vector<bm_tensor_t> present_key_cache, present_value_cache;
  std::vector<bm_tensor_t> inputs_lm, outputs_lm;
};

void Qwen::net_launch(const std::string &net_name,
                      std::vector<bm_tensor_t> &inputs,
                      std::vector<bm_tensor_t> &outputs, int stage_idx) {
  bool ret = bmrt_launch_tensor_ex(p_bmrt, net_name.c_str(), inputs.data(),
                                   inputs.size(), outputs.data(),
                                   outputs.size(), true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void Qwen::init(const std::vector<int> &devices,
                const std::string &model_path) {

  // request bm_handle
  device_num = devices.size();
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
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // embed, lm_head
  name_embed = "embedding";
  name_embed_cache = "embedding_cache";
  name_lm = "lm_head";
  net_embed = bmrt_get_network_info(p_bmrt, name_embed.c_str());
  net_embed_cache = bmrt_get_network_info(p_bmrt, name_embed_cache.c_str());
  net_lm = bmrt_get_network_info(p_bmrt, name_lm.c_str());
  int num_dims = net_embed->stages[0].input_shapes[0].num_dims;
  SEQLEN =
      net_embed->stages[0].input_shapes[0].dims[num_dims - 1]; // real seqlen
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 3) / 2;

  // blocks
  name_blocks.resize(NUM_LAYERS);
  name_blocks_cache.resize(NUM_LAYERS);
  net_blocks.resize(NUM_LAYERS);
  net_blocks_cache.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    name_blocks[i] = "block_" + std::to_string(i);
    name_blocks_cache[i] = "block_cache_" + std::to_string(i);
    net_blocks[i] = bmrt_get_network_info(p_bmrt, name_blocks[i].c_str());
    net_blocks_cache[i] =
        bmrt_get_network_info(p_bmrt, name_blocks_cache[i].c_str());
  }

  // net device mem

  hidden_states.resize(device_num);
  hidden_states_cache.resize(device_num);
  inputs_embed.resize(device_num);
  inputs_embed_cache.resize(device_num);
  int out_num = net_blocks[0]->output_num / device_num;
  int out_num_cache = net_blocks_cache[0]->output_num / device_num;
  for (int i = 0; i < device_num; ++i) {
    bmrt_tensor_with_device(
        &hidden_states[i],
        net_blocks[0]->stages[0].output_mems[0 + i * out_num],
        net_blocks[0]->output_dtypes[0 + i * out_num],
        net_blocks[0]->stages[0].output_shapes[0 + out_num]);
    bmrt_tensor_with_device(
        &hidden_states_cache[i],
        net_blocks_cache[0]->stages[0].output_mems[0 + i * out_num_cache],
        net_blocks_cache[0]->output_dtypes[0 + i * out_num_cache],
        net_blocks_cache[0]->stages[0].output_shapes[0 + out_num_cache]);
    bmrt_tensor_with_device(
        &inputs_embed[i], net_embed->stages[0].input_mems[i],
        net_embed->input_dtypes[i], net_embed->stages[0].input_shapes[i]);
    bmrt_tensor_with_device(&inputs_embed_cache[i],
                            net_embed_cache->stages[0].input_mems[i],
                            net_embed_cache->input_dtypes[i],
                            net_embed_cache->stages[0].input_shapes[i]);
  }

  inputs_pid.resize(device_num);
  inputs_attention.resize(device_num);
  next_pid.resize(device_num);
  next_attention.resize(device_num);
  int in_num = net_blocks[0]->input_num / device_num;
  int in_num_cache = net_blocks_cache[0]->input_num / device_num;
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

    ret = bmrt_tensor_ex(
        &next_pid[i], p_bmrt,
        net_blocks_cache[0]->input_loc_devices[1 + i * in_num_cache],
        net_blocks_cache[0]->input_dtypes[1 + i * in_num_cache],
        net_blocks_cache[0]->stages[0].input_shapes[1 + i * in_num_cache]);
    assert(true == ret);

    ret = bmrt_tensor_ex(
        &next_attention[i], p_bmrt,
        net_blocks_cache[0]->input_loc_devices[2 + i * in_num_cache],
        net_blocks_cache[0]->input_dtypes[2 + i * in_num_cache],
        net_blocks_cache[0]->stages[0].input_shapes[2 + i * in_num_cache]);
    assert(true == ret);
  }

  is_dynamic = net_blocks[0]->is_dynamic;
  past_keys.resize(NUM_LAYERS);
  past_values.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_keys[i].resize(device_num);
    past_values[i].resize(device_num);
    auto &net = net_blocks_cache[i];
    for (int j = 0; j < device_num; j++) {
      bmrt_tensor_with_device(
          &past_keys[i][j], net->stages[0].input_mems[3 + j * in_num_cache],
          net->input_dtypes[3 + j * in_num_cache],
          net->stages[0].input_shapes[3 + j * in_num_cache]);
      bmrt_tensor_with_device(
          &past_values[i][j], net->stages[0].input_mems[4 + j * in_num_cache],
          net->input_dtypes[4 + j * in_num_cache],
          net->stages[0].input_shapes[4 + j * in_num_cache]);
    }
  }

  int value = 0;
  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; j++) {
      bool status = bm_memset_device_ext(handles[j], &value, 1,
                                         past_keys[i][j].device_mem);
      assert(BM_SUCCESS == status);
      status = bm_memset_device_ext(handles[j], &value, 1,
                                    past_values[i][j].device_mem);
      assert(BM_SUCCESS == status);
    }
  }

  for (int j = 0; j < device_num; j++) {
    bool status = bm_memset_device_ext(handles[j], &value, 1,
                                       hidden_states[j].device_mem);
    assert(BM_SUCCESS == status);
    status = bm_memset_device_ext(handles[j], &value, 1,
                                  hidden_states_cache[j].device_mem);
    assert(BM_SUCCESS == status);
  }

  present_key_cache.resize(device_num);
  present_value_cache.resize(device_num);
  inputs_lm.resize(device_num);
  outputs_lm.resize(device_num);
  for (int i = 0; i < device_num; ++i) {
    present_key_cache[i] = past_keys[0][i];
    present_value_cache[i] = past_values[0][i];
    present_key_cache[i].shape.dims[1] = 1;
    present_value_cache[i].shape.dims[1] = 1;

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
    bm_free_device(handles[i], inputs_pid[i].device_mem);
    bm_free_device(handles[i], next_pid[i].device_mem);
    bm_free_device(handles[i], inputs_attention[i].device_mem);
    bm_free_device(handles[i], next_attention[i].device_mem);
    bm_free_device(handles[i], inputs_lm[i].device_mem);
    bm_free_device(handles[i], outputs_lm[i].device_mem);
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

int Qwen::forward_first(std::vector<int> &tokens) {
  std::vector<int> input_ids(SEQLEN, 0);
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, BF16_NEG_10000);
  std::copy(tokens.begin(), tokens.end(), input_ids.data());

  token_length = tokens.size();
  TOKEN_LEN = tokens.size();

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  if (is_dynamic) {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j < TOKEN_LEN; j++) {
        if (j <= i) {
          attention_mask[i * TOKEN_LEN + j] = 0;
        }
      }
    }
  } else {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j < SEQLEN; j++) {
        if (j <= i) {
          attention_mask[i * SEQLEN + j] = 0;
        }
      }
    }
  }

  // forward embeding
  std::vector<int> input_nums(device_num, 1);
  std::vector<void *> datas(device_num, (void *)input_ids.data());
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_embed.data(), datas.data(),
                           input_nums.data(), device_num);
  auto output_embeds = hidden_states;
  for (int i = 0; i < device_num; ++i) {
    output_embeds[i].shape = net_embed[0].stages[0].output_shapes[0];
  }
  auto ret = bmrt_launch_tensor_ex(
      p_bmrt, name_embed.c_str(), inputs_embed.data(), inputs_embed.size(),
      output_embeds.data(), output_embeds.size(), true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  std::vector<void *> pos_id_datas(device_num, (void *)position_id.data());
  std::vector<void *> in_attn_datas(device_num, (void *)attention_mask.data());
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_pid.data(), pos_id_datas.data(),
                           input_nums.data(), device_num);
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_attention.data(),
                           in_attn_datas.data(), input_nums.data(), device_num);
  auto tmp_hidden_states = hidden_states;
  std::vector<bm_tensor_t> inputs_block;
  std::vector<bm_tensor_t> outputs_block;
  for (int i = 0; i < device_num; ++i) {
    tmp_hidden_states[i].shape = net_blocks[0]->stages[0].input_shapes[0];
    inputs_block.push_back(tmp_hidden_states[i]);
    inputs_block.push_back(inputs_pid[i]);
    inputs_block.push_back(inputs_attention[i]);
    outputs_block.push_back(tmp_hidden_states[i]);
    outputs_block.push_back(past_keys[0][i]);
    outputs_block.push_back(past_values[0][i]);
    if (is_dynamic) {
      int h_bytes =
          bm_mem_get_device_size(tmp_hidden_states[i].device_mem) / SEQLEN;
      bm_set_device_mem(
          &inputs_block[0 + i * 3].device_mem, h_bytes * TOKEN_LEN,
          bm_mem_get_device_addr(tmp_hidden_states[i].device_mem));
      int pid_bytes = bm_mem_get_device_size(inputs_pid[i].device_mem) / SEQLEN;
      bm_set_device_mem(&inputs_block[1 + i * 3].device_mem,
                        pid_bytes * TOKEN_LEN,
                        bm_mem_get_device_addr(inputs_pid[i].device_mem));
      int mask_bytes = bm_mem_get_device_size(inputs_attention[i].device_mem) /
                       SEQLEN / SEQLEN;
      bm_set_device_mem(&inputs_block[2 + i * 3].device_mem,
                        mask_bytes * TOKEN_LEN * TOKEN_LEN,
                        bm_mem_get_device_addr(inputs_attention[i].device_mem));
      inputs_block[0 + i * 3].shape.dims[1] = TOKEN_LEN;
      inputs_block[1 + i * 3].shape.dims[1] = TOKEN_LEN;
      inputs_block[2 + i * 3].shape.dims[2] = TOKEN_LEN;
      inputs_block[2 + i * 3].shape.dims[3] = TOKEN_LEN;
    }
  }

  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; ++j) {
      outputs_block[1 + j * 3] = past_keys[i][j];
      outputs_block[2 + j * 3] = past_values[i][j];
    }
    net_launch(name_blocks[i], inputs_block, outputs_block);
  }

  int bytes = hidden_states[0].device_mem.size / SEQLEN;
  bm_memcpy_d2d_byte(bm_handle, inputs_lm[0].device_mem, 0,
                     hidden_states[0].device_mem, (token_length - 1) * bytes,
                     bytes);
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm[0], 1,
                              &outputs_lm[0], 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm[0].device_mem);
  return token;
}

int Qwen::forward_next(int cur_token) {
  token_length += 1;
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = BF16_NEG_10000;
  }
  int32_t position_id = token_length - 1;

  // embedding
  // std::vector<bm_tensor_t> inputs_embed;
  std::vector<void *> input_datas;
  std::vector<int> input_nums(device_num, 1);
  for (int i = 0; i < device_num; ++i) {
    // inputs_embed_cache.push_back(outputs_lm[i]); // token_id
    // inputs_embed_cache[i].shape = net_embed_cache->stages[0].input_shapes[0];
    input_datas.push_back((void *)(&cur_token));
  }
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_embed_cache.data(),
                           input_datas.data(), input_nums.data(), device_num);
  auto outputs_embed_cache = hidden_states_cache;
  for (int i = 0; i < device_num; ++i) {
    outputs_embed_cache[i].shape =
        net_embed_cache[0].stages[0].output_shapes[0];
  }
  auto ret = bmrt_launch_tensor_ex(
      p_bmrt, name_embed_cache.c_str(), inputs_embed_cache.data(),
      inputs_embed_cache.size(), outputs_embed_cache.data(),
      outputs_embed_cache.size(), true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // blocks
  std::vector<void *> attn_datas(device_num, attention_mask.data());
  std::vector<void *> pid_datas(device_num, &position_id);
  bmrt_memcpy_s2d_parallel(p_bmrt, next_attention.data(), attn_datas.data(),
                           input_nums.data(), device_num);
  bmrt_memcpy_s2d_parallel(p_bmrt, next_pid.data(), pid_datas.data(),
                           input_nums.data(), device_num);
  // WARNING: make inputs_lm device_num
  std::vector<bm_tensor_t> embed_1 = hidden_states_cache;
  for (int i = 0; i < device_num; ++i) {
    embed_1[i].shape = net_blocks_cache[0]->stages[0].input_shapes[0];
  }
  std::vector<bm_tensor_t> inputs_block;
  std::vector<bm_tensor_t> outputs_block;
  for (int i = 0; i < device_num; ++i) {
    inputs_block.push_back(embed_1[i]);
    inputs_block.push_back(next_pid[i]);
    inputs_block.push_back(next_attention[i]);
    inputs_block.push_back(past_keys[0][i]);
    inputs_block.push_back(past_values[0][i]);
    outputs_block.push_back(embed_1[i]);
    outputs_block.push_back(present_key_cache[i]);
    outputs_block.push_back(present_value_cache[i]);
  }

  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; ++j) {
      inputs_block[3 + j * 5] = past_keys[i][j];
      inputs_block[4 + j * 5] = past_values[i][j];
      int bytes = bm_mem_get_device_size(past_keys[0][j].device_mem) / SEQLEN;
      int token_offset = (token_length - 1) * bytes;
      bm_set_device_mem(&outputs_block[1 + j * 3].device_mem, bytes,
                        bm_mem_get_device_addr(past_keys[i][j].device_mem) +
                            token_offset);
      bm_set_device_mem(&outputs_block[2 + j * 3].device_mem, bytes,
                        bm_mem_get_device_addr(past_values[i][j].device_mem) +
                            token_offset);
    }
    net_launch(name_blocks_cache[i], inputs_block, outputs_block);
  }

  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &hidden_states_cache[0],
                              1, &outputs_lm[0], 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm[0].device_mem);
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen>(m, "Qwen")
      .def(pybind11::init<>())
      .def("init", &Qwen::init)
      .def("forward_first", &Qwen::forward_first)
      .def("forward_next", &Qwen::forward_next)
      .def("deinit", &Qwen::deinit)
      .def_readwrite("SEQLEN", &Qwen::SEQLEN) // read SEQLEN in pipeline.py
      .def_readwrite("token_length", &Qwen::token_length);
}
