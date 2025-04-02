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
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <vector>
#include "utils.h"
#include "cnpy.h"

template <typename T>
void dump_tensor_to_file(
        bm_handle_t&          handle,
        bm_device_mem_t&          t,
        std::vector<size_t>&& shape,
        const std::string&    filename,
        const std::string&    tensor_name) {
    int  cnt = bm_mem_get_device_size(t) / sizeof(T);
    auto buffer = std::make_unique<T[]>(cnt);
    bm_memcpy_d2s(handle, buffer.get(), t);
 
    if constexpr (std::is_same_v<T, uint16_t>) {
      std::vector<float> data(cnt);
      for (int i = 0; i < cnt; i++)
        // data[i] = bf16_to_fp32_value(buffer[i]);
        data[i] = fp16_ieee_to_fp32_value(buffer[i]);
      cnpy::npz_save(filename, tensor_name, data.data(), shape, "a");
    } else if constexpr (std::is_same_v<T, int32_t>){
      std::vector<int> data(cnt);
      memcpy(data.data(), buffer.get(), sizeof(int) * cnt);
      cnpy::npz_save(filename, tensor_name, data.data(), shape, "a");
    } else {
      std::vector<float> data(cnt);
      memcpy(data.data(), buffer.get(), sizeof(float) * cnt);
      cnpy::npz_save(filename, tensor_name, data.data(), shape, "a");
    }
}

// static const float MASK = 1;
// static const float MASK_CACHE = 0.0;
// uint16_t mask = fp32_to_fp16_bits(MASK);
// uint16_t mask_cache = fp32_to_fp16_bits(MASK_CACHE);
static const uint8_t mask = 1;
static const uint8_t mask_cache = 0;

class ChatGLM {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  int forward_first(std::vector<int> &tokens, std::vector<float> &images, int begin, int end);
  int forward_next();
  std::vector<int> generate(std::vector<int> &history_tokens, std::vector<float> &images, int begin, int end, int EOS);

  std::mt19937 sgen;
  ChatGLM() : sgen(std::random_device()()){};

private:
  void move2end(const bm_device_mem_t &kv);
  void move_kv_cache(
    bm_tensor_t &past_key_v,
    bm_tensor_t &past_value_v,
    std::vector<bm_tensor_t> &outputs_block,
    int token_length
  );
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  std::vector<std::vector<bm_tensor_t>>  tensor_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch(const bm_net_info_t *net, std::vector<bm_tensor_t> &in_tensors, std::vector<bm_tensor_t> &out_tensors);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);

  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

public:
  int token_length;
  int SEQLEN;     // read from bmodel
  int NUM_LAYERS; // read from bmodel
  int HIDDEN_SIZE;
  bool io_alone;
  int device_num;
  std::vector<int> visited_tokens;

  // generation
  float temperature;
  float top_p;
  float repeat_penalty;
  int repeat_last_n;
  int max_new_tokens;
  std::string generation_mode;
  std::string prompt_mode;

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_penalty_sample_head;
  const bm_net_info_t *net_vit;
  std::vector<bm_tensor_t> past_key, past_value;
};

void ChatGLM::net_launch(const bm_net_info_t *net, int stage_idx) {
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

std::vector<std::vector<bm_tensor_t>> 
ChatGLM::tensor_launch(const bm_net_info_t *net, int stage_idx) {
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
  return {in_tensors, out_tensors};
}

void ChatGLM::net_launch(const bm_net_info_t *net, std::vector<bm_tensor_t> &in_tensors, std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void ChatGLM::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void ChatGLM::init(const std::vector<int> &devices, std::string model_path) {

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
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  net_penalty_sample_head = bmrt_get_network_info(p_bmrt, "penalty_sample_head");
  net_vit = bmrt_get_network_info(p_bmrt, "glm4v_vit");
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 5) / 2;

  // resize
  visited_tokens.resize(SEQLEN, 0);

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  // kv cache
  auto addr_mode = net_blocks_cache[0]->addr_mode;
  io_alone = (addr_mode == 1);
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  if (io_alone) {
    for (int i = 0; i < NUM_LAYERS; i++) {
      auto &net = net_blocks_cache[i];
      bmrt_tensor_with_device(
        &past_key[i],
        net->stages[0].input_mems[3],
        net->input_dtypes[3],
        net->stages[0].input_shapes[3]);
      bmrt_tensor_with_device(
        &past_value[i],
        net->stages[0].input_mems[4],
        net->input_dtypes[4],
        net->stages[0].input_shapes[4]);
    }
  } else {
    for (int i = 0; i < NUM_LAYERS; i++) {
      auto &net = net_blocks_cache[i];
      ret = bmrt_tensor_ex(
        &past_key[i], p_bmrt,
        net->input_loc_devices[3],
        net->input_dtypes[3],
        net->stages[0].input_shapes[3]);
      assert(true == ret);
      ret = bmrt_tensor_ex(
        &past_value[i], p_bmrt,
        net->input_loc_devices[4],
        net->input_dtypes[4],
        net->stages[0].input_shapes[4]);
      assert(true == ret);
    }
  }
}

void ChatGLM::deinit() {
  if (false == io_alone) {
    for (int i = 0; i < NUM_LAYERS; i++) {
      bm_free_device(bm_handle, past_key[i].device_mem);
      bm_free_device(bm_handle, past_value[i].device_mem);
    }
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

void ChatGLM::move_kv_cache(
    bm_tensor_t &past_key_v,
    bm_tensor_t &past_value_v,
    std::vector<bm_tensor_t> &outputs_block,
    int token_length) {
  std::vector<bm_tensor_t> dst_tensors, src_tensors;
  dst_tensors.push_back(past_key_v);
  dst_tensors.push_back(past_value_v); 
  src_tensors.push_back(outputs_block[1]);
  src_tensors.push_back(outputs_block[2]);

  std::vector<bm_shape_t> shapes, dst_strides, src_strides;
  bm_shape_t shape = past_key_v.shape;
  bm_shape_t dst_str = {
    .num_dims = 4,
    .dims = {
      shape.dims[1]*shape.dims[2]*shape.dims[3],
      shape.dims[2]*shape.dims[3],
      shape.dims[3],
      1
    }
  };
  // key_stride, value_stride
  dst_strides.push_back(dst_str);
  dst_strides.push_back(dst_str);

  shape.dims[2] = 1;
  shapes.push_back(shape);
  shapes.push_back(shape);
  bm_shape_t src_str = {
    .num_dims = 4,
    .dims = {
      shape.dims[1]*shape.dims[2]*shape.dims[3],
      shape.dims[2]*shape.dims[3],
      shape.dims[3],
      1
    }
  };
  // key_stride, value_stride
  src_strides.push_back(src_str);
  src_strides.push_back(src_str);

  std::vector<int> tensor_num(device_num, 2);
  std::vector<size_t> dst_offsets, src_offsets(device_num * 2, 0);
  int bytes = bm_mem_get_device_size(past_key_v.device_mem) / SEQLEN / shape.dims[1];    // 1*2*512*128/512/2
  int token_offset = token_length * bytes;
  dst_offsets.push_back(token_offset);
  dst_offsets.push_back(token_offset);
  bool ret = bmrt_memcpy_d2d_stride_ex_parallel(
      p_bmrt, dst_tensors.data(), dst_offsets.data(), dst_strides.data(),
      src_tensors.data(), src_offsets.data(), src_strides.data(),
      shapes.data(), tensor_num.data(), device_num);
  assert(ret);
}

void ChatGLM::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
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

int ChatGLM::greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int ChatGLM::penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  auto &in1_mem = net->stages[0].input_mems[1];
  auto &in2_mem = net->stages[0].input_mems[2];
  auto &in3_mem = net->stages[0].input_mems[3];
  auto &in4_mem = net->stages[0].input_mems[4];
  auto &out0_mem = net->stages[0].output_mems[0];
  auto &out1_mem = net->stages[0].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  std::vector<int> generated_tokens(SEQLEN, visited_tokens[token_length - 1]);
  repeat_last_n = std::min(repeat_last_n, token_length);
  std::copy(visited_tokens.begin() + token_length - repeat_last_n, 
            visited_tokens.begin() + token_length,
            generated_tokens.begin());
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)generated_tokens.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, in4_mem, (void *)&repeat_penalty);

  // inference
  head_launch(net, logits_mem);

  // get logit & token
  int candidate_num = net->stages[0].output_shapes[0].dims[1];
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s(bm_handle, probs.data(), out0_mem);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s(bm_handle, tokens.data(), out1_mem);

  // penalty_sample
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int ChatGLM::forward_first(std::vector<int> &tokens, std::vector<float> &images, int begin, int end) {
  std::vector<int> position_id(SEQLEN, 0);
  //std::vector<bool> attention_mask(SEQLEN * SEQLEN, false);
  std::vector<uint8_t> attention_mask(SEQLEN * SEQLEN, 0);
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.begin());

  token_length = tokens.size();

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < SEQLEN; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i) {
      } else {
        attention_mask[i * SEQLEN + j] = mask;
      }
    }
  }

  // forward embeding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  net_launch(net_embed); // prefil embedding

   // forward vision transformer
  auto &vit_in_mem = net_vit->stages[0].input_mems[0];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, vit_in_mem, (void *)images.data());
  net_launch(net_vit);

// concatenate texting embedding and image embedding
  int dst_offset = begin * HIDDEN_SIZE * 2;
  int size = (end - begin - 1) * HIDDEN_SIZE * 2;
  bm_memcpy_d2d_byte(bm_handle, out_mem, dst_offset, vit_out_mem, 0, size);

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx].device_mem, net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx].device_mem, net_blocks[idx]->stages[0].output_mems[2]);
  }

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
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

int ChatGLM::forward_next() {
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint8_t> attention_mask(SEQLEN + 1, 0);
  for (int i = 0; i <= SEQLEN - token_length; i++) {
    attention_mask[i] = mask_cache;
  }
  int32_t position_id = token_length - 1;
  // embedding
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(net_embed_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (io_alone) {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      } else {
        d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
      }
    } else {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      in3_mem = past_key[idx].device_mem;
      in4_mem = past_value[idx].device_mem;
    }
    auto tensors = tensor_launch(net_blocks_cache[idx]);
    auto in_tensors = tensors[0];
    auto out_tensors = tensors[1];
    net_launch(net_blocks_cache[idx], in_tensors, out_tensors);
    out_mem = out0_mem;
    move_kv_cache(past_key[idx], past_value[idx], out_tensors, token_length);

    // if (idx == 0 && position_id < 260) {
    //   dump_tensor_to_file<uint16_t>(bm_handle,in0_mem, {1,1,4096},   "input_states.npz",    "input_cache_" + std::to_string(position_id));
    //   dump_tensor_to_file<int32_t>(bm_handle,in1_mem,  {1,1},        "position_ids.npz",    "input_cache_" + std::to_string(position_id));
    //   dump_tensor_to_file<uint16_t>(bm_handle,in2_mem, {1,1,1,513},  "attention_mask.npz",  "input_cache_" + std::to_string(position_id));
    //   dump_tensor_to_file<uint16_t>(bm_handle,in3_mem, {1,2,512,128},"history_k.npz",       "input_cache_" + std::to_string(position_id));
    //   dump_tensor_to_file<uint16_t>(bm_handle,in4_mem, {1,2,512,128},"history_v.npz",       "input_cache_" + std::to_string(position_id));
    //   dump_tensor_to_file<uint16_t>(bm_handle,out0_mem,{1,1,4096},   "hidden_states.npz",   "input_cache_" + std::to_string(position_id));
    //   dump_tensor_to_file<uint16_t>(bm_handle,out1_mem,{1,2,1,128},  "past_k.npz",          "input_cache_" + std::to_string(position_id));
    //   dump_tensor_to_file<uint16_t>(bm_handle,out2_mem,{1,2,1,128},  "past_v.npz",          "input_cache_" + std::to_string(position_id));
    // }
  }

  // forward lmhead
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm);

  int token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

std::vector<int> ChatGLM::generate(std::vector<int> &history_tokens, std::vector<float> &images, int begin, int end, int EOS) {
  if (history_tokens.empty()) {
    printf("Sorry: your question is empty!!\n");
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
  int token = forward_first(history_tokens, images, begin, end);
  while (token != EOS && token_length < SEQLEN) {
    result_tokens.emplace_back(token);
    token = forward_next();
  }

  return result_tokens;
}


PYBIND11_MODULE(chat, m) {
  pybind11::class_<ChatGLM>(m, "ChatGLM")
      .def(pybind11::init<>())
      .def("init", &ChatGLM::init)
      .def("forward_first", &ChatGLM::forward_first)
      .def("forward_next", &ChatGLM::forward_next)
      .def("generate", &ChatGLM::generate)
      .def("deinit", &ChatGLM::deinit)
      .def_readwrite("SEQLEN", &ChatGLM::SEQLEN) // read SEQLEN in pipeline.py
      .def_readwrite("token_length", &ChatGLM::token_length)
      .def_readwrite("temperature", &ChatGLM::temperature)
      .def_readwrite("top_p", &ChatGLM::top_p)
      .def_readwrite("repeat_penalty", &ChatGLM::repeat_penalty)
      .def_readwrite("repeat_last_n", &ChatGLM::repeat_last_n)
      .def_readwrite("max_new_tokens", &ChatGLM::max_new_tokens)
      .def_readwrite("generation_mode", &ChatGLM::generation_mode)
      .def_readwrite("prompt_mode", &ChatGLM::prompt_mode);
}
