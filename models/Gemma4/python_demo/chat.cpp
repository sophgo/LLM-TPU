//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
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
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <set>
#include <stdio.h>
#include <vector>

namespace py = pybind11;
using ArrayFloat = py::array_t<float, py::array::c_style | py::array::forcecast>;
using ArrayInt = py::array_t<int, py::array::c_style | py::array::forcecast>;

//===------------------------------------------------------------===//
// Empty Func
//===------------------------------------------------------------===//
void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
}

void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
               int stage_idx = 0) {
  for (int i = 0; i < net->input_num; i++) {
    empty(bm_handle, net->stages[stage_idx].input_mems[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    empty(bm_handle, net->stages[stage_idx].output_mems[i]);
  }
}

class Gemma4 {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void forward_embed(std::vector<int> &tokens);
  void forward_vit(
      py::array_t<float, py::array::c_style | py::array::forcecast> const
          &pixel_values,
      py::array_t<int, py::array::c_style | py::array::forcecast> const
          &pixel_position_ids,
      std::vector<int> vit_offsets,
      int mm_tokens_actual,
      bool is_video = false);
  void forward_audio(ArrayFloat const &audio_features,
                     ArrayInt const &audio_offset);
    int forward_first();
  int forward_next();

  std::mt19937 sgen;
  Gemma4() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
                  int size = 0);
  void init_by_names();
  int greedy_search(bm_device_mem_t &logits_mem);
  int penalty_sample(bm_device_mem_t &logits_mem);

public:
  int hidden_bytes;
  int token_length;
  int SEQLEN; // KV cache length (from block_cache shapes)
  int MAX_INPUT_LENGTH; // max prefill length (from net_embed shapes)
  int HIDDEN_SIZE;
  int NUM_LAYERS; // read from bmodel
  int SLIDING_WINDOW; // sliding window size for sliding_attention layers
  int ID_IMAGE_PAD;
  int ID_VIDEO_PAD;
  int ID_AUDIO_PAD;
  int ID_BOA;          // beginning of audio marker
  int ID_EOA;          // end of audio marker
  int AUDIO_MEL;      // max mel frames (from audio bmodel input shape)
  int AUDIO_TOKENS;   // max audio tokens (from audio bmodel output shape)
  std::vector<int> cur_input_ids;
  bool lmhead_with_topk;
  bool is_dynamic;
  std::vector<int> visited_tokens;
  uint16_t mask_value;

  // Gemma4-specific: KV-shared and mixed head_dim
  std::vector<bool> is_shared_layer;
  std::vector<bool> is_full_layer;
  std::vector<int> kv_shared_source; // source layer idx for shared, -1 for normal
  std::vector<int> kv_bytes_per_token; // per-layer kv bytes (varies by head_dim)

  // generation
  std::string generation_mode;
  float penalty;
  float temperature;
  int top_k;
  float top_p;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_vit_image;
  const bm_net_info_t *net_vit_video;
  const bm_net_info_t *net_audio;
  bm_device_mem_t dev_buffer;
  bm_device_mem_t dev_buffer_cache;
  bm_device_mem_t dev_buffer_ids;       // persistent buffer for input_ids (prefill)
  bm_device_mem_t dev_buffer_ids_cache; // persistent buffer for cur_token (decode)
  bm_device_mem_t dev_position_ids;     // persistent buffer for position_ids (prefill)
  bm_device_mem_t dev_position_ids_cache; // persistent buffer for position_ids (decode)
  bm_device_mem_t dev_mask_full;        // persistent buffer for full_attention_mask (prefill)
  bm_device_mem_t dev_mask_sliding;     // persistent buffer for sliding_attention_mask (prefill)
  bm_device_mem_t dev_mask_full_cache;      // full mask for normal decode layers
  bm_device_mem_t dev_mask_full_shared_cache; // full mask for shared decode layers
  bm_device_mem_t dev_mask_sliding_cache;     // sliding mask for normal decode layers
  bm_device_mem_t dev_mask_sliding_shared_cache; // sliding mask for shared decode layers
  const bm_net_info_t *net_lm, *net_greedy_head, *net_sample_head;
  std::vector<bm_device_mem_t> past_key;   // allocated for normal layers only
  std::vector<bm_device_mem_t> past_value; // allocated for normal layers only
};

void Gemma4::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                 int size) {
  if (!size)
    size = bm_mem_get_device_size(src);
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Gemma4::init_by_names() {
  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0) {
        return true;
      }
    }
    return false;
  };

  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
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
  net_vit_image = nullptr;
  if (is_exist("vit_image", net_names, num_nets)) {
    net_vit_image = bmrt_get_network_info(p_bmrt, "vit_image");
  }
  net_vit_video = nullptr;
  if (is_exist("vit_video", net_names, num_nets)) {
    net_vit_video = bmrt_get_network_info(p_bmrt, "vit_video");
  }
  net_audio = nullptr;
  if (is_exist("audio", net_names, num_nets)) {
    net_audio = bmrt_get_network_info(p_bmrt, "audio");
  }

  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1]; // max prefill length
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;

  // Count block nets (exclude embed, embedding_cache, lm_head, greedy_head,
  // sample_head, vit)
  int extra_nets = 2; // embed + lm_head
  if (is_exist("embedding_cache", net_names, num_nets))
    extra_nets++;
  if (net_greedy_head)
    extra_nets++;
  if (net_sample_head)
    extra_nets++;
  if (net_vit_image)
    extra_nets++;
  if (net_vit_video)
    extra_nets++;
  if (net_audio)
    extra_nets++;
  auto num_blocks = num_nets - extra_nets;

  NUM_LAYERS = num_blocks / 2; // 2 nets per block: block + block_cache
  for (int i = 0; i < num_blocks / 2; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    if ((!is_exist(block_name.c_str(), net_names, num_nets)) ||
        (!is_exist(cache_name.c_str(), net_names, num_nets))) {
      NUM_LAYERS = i;
      printf("Warning: Only %d blocks found, expected %d blocks.\n", NUM_LAYERS,
             num_blocks / 2);
      break;
    }
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }
  free(net_names);

  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[5].dims[1]; // KV cache length

  if (net_audio) {
    AUDIO_MEL = net_audio->stages[0].input_shapes[0].dims[2];
    AUDIO_TOKENS = net_audio->stages[0].output_shapes[0].dims[1];
  }

  // Determine mask_value from dtype
  if (net_blocks[0]->input_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2;
  } else if (net_blocks[0]->input_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C;
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }

  // Determine shared/full layer properties from block_cache net structure
  // Shared layer: 1 output (output_states only)
  // Normal layer: 3 outputs (output_states, k_cache, v_cache)
  is_shared_layer.resize(NUM_LAYERS);
  is_full_layer.resize(NUM_LAYERS);
  kv_bytes_per_token.resize(NUM_LAYERS);
  kv_shared_source.resize(NUM_LAYERS);

  for (int i = 0; i < NUM_LAYERS; i++) {
    is_shared_layer[i] = (net_blocks_cache[i]->output_num == 1);

    if (!is_shared_layer[i]) {
      // Normal layer: k_cache output shape [1, 1, num_kv_heads, cur_head_dim]
      auto &kv_shape = net_blocks_cache[i]->stages[0].output_shapes[1];
      int cur_head_dim = kv_shape.dims[3];
      is_full_layer[i] = (cur_head_dim > 256); // full=512, sliding=256
      kv_bytes_per_token[i] =
          bm_mem_get_device_size(net_blocks_cache[i]->stages[0].output_mems[1]);
    } else {
      // Shared layer: shared_k input shape [1, SEQLEN+1, num_kv_heads,
      // cur_head_dim]
      auto &shared_k_shape = net_blocks_cache[i]->stages[0].input_shapes[5];
      int cur_head_dim = shared_k_shape.dims[3];
      is_full_layer[i] = (cur_head_dim > 256);
      // kv_bytes_per_token set below after source layers are known
    }
  }

  // Compute kv_shared_source using HF algorithm:
  // Find last normal layer of same attention type
  int last_normal_sliding = -1;
  int last_normal_full = -1;
  for (int i = 0; i < NUM_LAYERS; i++) {
    if (!is_shared_layer[i]) {
      if (is_full_layer[i]) {
        last_normal_full = i;
      } else {
        last_normal_sliding = i;
      }
    }
  }

  for (int i = 0; i < NUM_LAYERS; i++) {
    if (is_shared_layer[i]) {
      kv_shared_source[i] =
          is_full_layer[i] ? last_normal_full : last_normal_sliding;
      kv_bytes_per_token[i] = kv_bytes_per_token[kv_shared_source[i]];
    } else {
      kv_shared_source[i] = -1; // normal layer, independent kv
    }
  }
}

void Gemma4::init(int dev_id, std::string model_path) {
  std::cout << "Device [" << dev_id << "] loading ....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  assert(BM_SUCCESS == status);

  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = false;
  ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  init_by_names();

  visited_tokens.resize(SEQLEN);
  hidden_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[0]);
  is_dynamic = net_blocks[0]->is_dynamic;

  // Allocate dev_buffer for embedding output
  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);
  auto buffer_size_cache =
      bm_mem_get_device_size(net_embed_cache->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer_cache, buffer_size_cache);
  assert(BM_SUCCESS == status); 

  // Allocate dev_buffer_ids for persistent input_ids storage
  auto ids_size = bm_mem_get_device_size(net_embed->stages[0].input_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer_ids, ids_size);
  assert(BM_SUCCESS == status);

  // Allocate dev_buffer_ids_cache for persistent cur_token storage (decode)
  auto ids_cache_size =
      bm_mem_get_device_size(net_embed_cache->stages[0].input_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer_ids_cache,
                                 ids_cache_size);
  assert(BM_SUCCESS == status);

  // Allocate persistent device buffers for reusable inputs (prefill)
  auto pos_size = bm_mem_get_device_size(net_blocks[0]->stages[0].input_mems[1]);
  status = bm_malloc_device_byte(bm_handle, &dev_position_ids, pos_size);
  assert(BM_SUCCESS == status);
  auto mask_size = bm_mem_get_device_size(net_blocks[0]->stages[0].input_mems[2]);
  status = bm_malloc_device_byte(bm_handle, &dev_mask_full, mask_size);
  assert(BM_SUCCESS == status);
  status = bm_malloc_device_byte(bm_handle, &dev_mask_sliding, mask_size);
  assert(BM_SUCCESS == status);

  // Allocate persistent device buffers for reusable inputs (decode)
  auto pos_cache_size = bm_mem_get_device_size(net_blocks_cache[0]->stages[0].input_mems[1]);
  status = bm_malloc_device_byte(bm_handle, &dev_position_ids_cache, pos_cache_size);
  assert(BM_SUCCESS == status);
  auto mask_cache_size = bm_mem_get_device_size(net_blocks_cache[0]->stages[0].input_mems[2]);
  status = bm_malloc_device_byte(bm_handle, &dev_mask_full_cache, mask_cache_size);
  assert(BM_SUCCESS == status);
  status = bm_malloc_device_byte(bm_handle, &dev_mask_full_shared_cache, mask_cache_size);
  assert(BM_SUCCESS == status);
  status = bm_malloc_device_byte(bm_handle, &dev_mask_sliding_cache, mask_cache_size);
  assert(BM_SUCCESS == status);
  status = bm_malloc_device_byte(bm_handle, &dev_mask_sliding_shared_cache, mask_cache_size);
  assert(BM_SUCCESS == status);

  // KV cache - allocate (SEQLEN+1) capacity so shared layers can reference
  // source layer's buffer with SEQLEN+1 shape requirement
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    if (!is_shared_layer[i]) {
      auto kv_size = (SEQLEN + 1) * kv_bytes_per_token[i];
      auto status = bm_malloc_device_byte(bm_handle, &past_key[i], kv_size);
      assert(BM_SUCCESS == status);
      status = bm_malloc_device_byte(bm_handle, &past_value[i], kv_size);
      assert(BM_SUCCESS == status);
      empty(bm_handle, past_key[i]);
      empty(bm_handle, past_value[i]);
    } else {
      int source = kv_shared_source[i];
      past_key[i] = past_key[source];
      past_value[i] = past_value[source];
    }
  }
}

void Gemma4::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bm_free_device(bm_handle, dev_buffer_cache);
  bm_free_device(bm_handle, dev_buffer_ids);
  bm_free_device(bm_handle, dev_buffer_ids_cache);
  bm_free_device(bm_handle, dev_position_ids);
  bm_free_device(bm_handle, dev_position_ids_cache);
  bm_free_device(bm_handle, dev_mask_full);
  bm_free_device(bm_handle, dev_mask_sliding);
  bm_free_device(bm_handle, dev_mask_full_cache);
  bm_free_device(bm_handle, dev_mask_full_shared_cache);
  bm_free_device(bm_handle, dev_mask_sliding_cache);
  bm_free_device(bm_handle, dev_mask_sliding_shared_cache);
  for (int i = 0; i < NUM_LAYERS; i++) {
    if (!is_shared_layer[i]) {
      bm_free_device(bm_handle, past_key[i]);
      bm_free_device(bm_handle, past_value[i]);
    }
  }
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void Gemma4::net_launch(const bm_net_info_t *net, int stage_idx) {
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
}

void Gemma4::forward_embed(std::vector<int> &tokens) {
  token_length = tokens.size();
  std::vector<int> input_ids(MAX_INPUT_LENGTH, 0);
  std::copy(tokens.begin(), tokens.end(), input_ids.data());
  for (int i = 0; i < token_length; i++) {
    if (input_ids[i] == ID_IMAGE_PAD || input_ids[i] == ID_VIDEO_PAD || input_ids[i] == ID_AUDIO_PAD) {
      input_ids[i] = 0;
    }
  }

  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)input_ids.data());
  net_launch(net_embed);
  d2d(dev_buffer, out_mem);
  d2d(dev_buffer_ids, in_mem);
  cur_input_ids = tokens;
}

void Gemma4::forward_vit(
    py::array_t<float, py::array::c_style | py::array::forcecast> const
        &pixel_values,
    py::array_t<int, py::array::c_style | py::array::forcecast> const
        &pixel_position_ids,
    std::vector<int> vit_offsets,
    int mm_tokens_actual,
    bool is_video) {
  auto net_vit = is_video ? net_vit_video : net_vit_image;
  assert(net_vit != nullptr);

  auto &vit_in_mem = net_vit->stages[0].input_mems[0];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  auto pv_buf = pixel_values.request();
  float *pv_ptr = static_cast<float *>(pv_buf.ptr);
  bm_memcpy_s2d(bm_handle, vit_in_mem, (void *)pv_ptr);

  auto &pos_in_mem = net_vit->stages[0].input_mems[1];
  auto pos_buf = pixel_position_ids.request();
  int *pos_ptr = static_cast<int *>(pos_buf.ptr);
  bm_memcpy_s2d(bm_handle, pos_in_mem, (void *)pos_ptr);

  net_launch(net_vit);

  // Copy vit output into dev_buffer at each frame's offset.
  // mm_tokens_actual: actual number of mm tokens in prompt (may be less than
  // bmodel's mm_tokens due to aspect-ratio-preserving resize).
  // Image: copy mm_tokens_actual tokens from [1, mm_tokens, H] output.
  // Video: copy mm_tokens_actual tokens per frame from [batch, mm_tokens, H].
  auto vit_out_shape = net_vit->stages[0].output_shapes[0];
  int batch_size = vit_out_shape.dims[0];
  int mm_tokens = vit_out_shape.dims[1];
  int one_bmodel_frame_bytes = mm_tokens * HIDDEN_SIZE * sizeof(uint16_t);
  int copy_bytes = mm_tokens_actual * HIDDEN_SIZE * sizeof(uint16_t);

  if (!is_video) {
    // Image: single contiguous copy of actual tokens
    int dst_offset = vit_offsets[0] * HIDDEN_SIZE * sizeof(uint16_t);
    bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset, vit_out_mem, 0,
                       copy_bytes);
  } else {
    // Video: per-frame copy (frames may have non-contiguous offsets in prompt)
    for (int f = 0; f < batch_size; f++) {
      int dst_offset = vit_offsets[f] * HIDDEN_SIZE * sizeof(uint16_t);
      int src_offset = f * one_bmodel_frame_bytes;
      bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset, vit_out_mem,
                         src_offset, copy_bytes);
    }
  }
}

void Gemma4::forward_audio(ArrayFloat const &audio_features,
                           ArrayInt const &audio_offset) {
  assert(net_audio != nullptr);

  auto p_audio_features = audio_features.request();
  auto p_audio = static_cast<float *>(p_audio_features.ptr);
  auto p_audio_offset = audio_offset.request();
  auto p_offsets = static_cast<int *>(p_audio_offset.ptr);
  int num_audio = audio_offset.size();
  int actual_audio_seq_len = p_offsets[num_audio - 1]; // last element is actual token count

  auto &audio_in_mem = net_audio->stages[0].input_mems[0];
  auto &audio_out_mem = net_audio->stages[0].output_mems[0];

  // Compute per-token bytes from bmodel shapes to detect dtype mismatch
  auto audio_out_total_size = bm_mem_get_device_size(audio_out_mem);
  int audio_out_tokens = net_audio->stages[0].output_shapes[0].dims[1];
  int audio_per_token_bytes = audio_out_total_size / audio_out_tokens;

  auto embed_out_total_size = bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  int embed_max_len = net_embed->stages[0].output_shapes[0].dims[1];
  int embed_per_token_bytes = embed_out_total_size / embed_max_len;

  if (audio_per_token_bytes != embed_per_token_bytes) {
    printf("ERROR: audio output per-token bytes (%d) != embedding per-token bytes (%d)\n",
           audio_per_token_bytes, embed_per_token_bytes);
    printf("  audio dtype may not match text model dtype, copy would produce wrong data!\n");
  }

  // Copy padded mel features to bmodel input
  bm_memcpy_s2d(bm_handle, audio_in_mem, (void *)p_audio);
  net_launch(net_audio);

  
  // Copy only actual_audio_seq_len tokens from output to dev_buffer
  // Use embed_per_token_bytes for destination offset (dev_buffer layout)
  // Use audio_per_token_bytes for source copy size if dtypes match
  int audio_start = p_offsets[0]; // first element is start offset in prompt
  int per_token_dst = embed_per_token_bytes; // bytes per token in dev_buffer
  int per_token_src = audio_per_token_bytes; // bytes per token in audio output
  if (per_token_src == per_token_dst) {
    // Same dtype: simple byte copy
    int copy_bytes = actual_audio_seq_len * per_token_dst;
    int dst_offset = audio_start * per_token_dst;
    bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset, audio_out_mem, 0,
                       copy_bytes);
  } else {
    // Dtype mismatch: copy what we can but data will be wrong
    // This should not happen if compile_audio uses same quantize as text model
    printf("WARNING: dtype mismatch, skipping audio embedding merge\n");
  }
}

int Gemma4::greedy_search(bm_device_mem_t &logits_mem) {
  auto &out_mem = net_greedy_head->stages[0].output_mems[0];
  bm_set_device_mem(&net_greedy_head->stages[0].input_mems[0], logits_mem.size,
                    logits_mem.u.device.device_addr);
  net_launch(net_greedy_head);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Gemma4::penalty_sample(bm_device_mem_t &logits_mem) {
  auto &in0_mem = net_sample_head->stages[0].input_mems[0];
  auto &in1_mem = net_sample_head->stages[0].input_mems[1];
  auto &in2_mem = net_sample_head->stages[0].input_mems[2];
  auto &in3_mem = net_sample_head->stages[0].input_mems[3];
  auto &in4_mem = net_sample_head->stages[0].input_mems[4];
  auto &in5_mem = net_sample_head->stages[0].input_mems[5];
  auto &out0_mem = net_sample_head->stages[0].output_mems[0];
  auto &out1_mem = net_sample_head->stages[0].output_mems[1];

  bm_memcpy_s2d(bm_handle, in1_mem, (void *)visited_tokens.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)&penalty);
  bm_memcpy_s2d(bm_handle, in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, in4_mem, (void *)&top_k);
  bm_memcpy_s2d(bm_handle, in5_mem, (void *)&top_p);

  d2d(in0_mem, logits_mem, 0, bm_mem_get_device_size(logits_mem));
  net_launch(net_sample_head);

  int candidate_num = top_k;
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s_partial_offset(bm_handle, probs.data(), out0_mem,
                               top_k * sizeof(float), 0);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s_partial_offset(bm_handle, tokens.data(), out1_mem,
                               top_k * sizeof(float), 0);

  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int Gemma4::forward_first() {
  // Mask stride: dynamic uses token_length, non-dynamic uses MAX_INPUT_LENGTH
  int mask_stride = is_dynamic ? token_length : MAX_INPUT_LENGTH;
  int mask_size = token_length * mask_stride;

  std::vector<int> position_id(is_dynamic ? token_length : MAX_INPUT_LENGTH, 0);
  std::vector<uint16_t> full_attention_mask(mask_size, mask_value);
  std::vector<uint16_t> sliding_attention_mask(mask_size, mask_value);

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  // Full causal mask: position i can see all positions j <= i
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j <= i; j++) {
      full_attention_mask[i * mask_stride + j] = 0;
    }
  }
  // Sliding window mask: position i can only see positions within window
  int window = SLIDING_WINDOW;
  for (int i = 0; i < token_length; i++) {
    for (int j = std::max(0, i - window + 1); j <= i; j++) {
      sliding_attention_mask[i * mask_stride + j] = 0;
    }
  }

  // Write reusable inputs to device buffers
  if (is_dynamic) {
    // Dynamic: mask is token_length × token_length, no memset needed
    bm_memcpy_s2d_partial(bm_handle, dev_position_ids, (void *)position_id.data(),
                          token_length * sizeof(int));
    bm_memcpy_s2d_partial(bm_handle, dev_mask_full, (void *)full_attention_mask.data(),
                          mask_size * sizeof(uint16_t));
    bm_memcpy_s2d_partial(bm_handle, dev_mask_sliding, (void *)sliding_attention_mask.data(),
                          mask_size * sizeof(uint16_t));
  } else {
    // Non-dynamic: memset mask buffers to mask_value, then partial write valid rows
    bm_memset_device_ext(bm_handle, &mask_value, sizeof(uint16_t), dev_mask_full);
    bm_memset_device_ext(bm_handle, &mask_value, sizeof(uint16_t), dev_mask_sliding);
    bm_memcpy_s2d(bm_handle, dev_position_ids, (void *)position_id.data());
    bm_memcpy_s2d_partial(bm_handle, dev_mask_full, (void *)full_attention_mask.data(),
                          mask_size * sizeof(uint16_t));
    bm_memcpy_s2d_partial(bm_handle, dev_mask_sliding, (void *)sliding_attention_mask.data(),
                          mask_size * sizeof(uint16_t));
  }

  // Empty block nets and past_key/past_value
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty_net(bm_handle, net_blocks[i]);
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    if (!is_shared_layer[i]) {
      empty(bm_handle, past_key[i]);
      empty(bm_handle, past_value[i]);
    }
  }

  // Forward blocks
  auto hidden_state = net_blocks[0]->stages[0].output_mems[0];
  int valid_bytes = token_length * hidden_bytes;
  if (is_dynamic) {
    d2d(hidden_state, dev_buffer, 0, valid_bytes);
  } else {
    d2d(hidden_state, dev_buffer);
  }
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &net = net_blocks[idx];
    int input_num = net->input_num;
    int output_num = net->output_num;

    std::vector<bm_tensor_t> in_tensors(input_num);
    std::vector<bm_tensor_t> out_tensors(output_num);

    // input[0]: input_states - d2d copy to each block's own memory
    auto &in0_mem = net->stages[0].input_mems[0];
    empty(bm_handle, in0_mem);
    if (is_dynamic) {
      d2d(in0_mem, hidden_state, 0, valid_bytes);
    } else {
      d2d(in0_mem, hidden_state);
    }
    bmrt_tensor_with_device(&in_tensors[0], in0_mem, net->input_dtypes[0],
                            net->stages[0].input_shapes[0]);

    // input[1]: position_ids - from persistent device buffer
    bmrt_tensor_with_device(&in_tensors[1], dev_position_ids,
                            net->input_dtypes[1],
                            net->stages[0].input_shapes[1]);

    // input[2]: attention_mask - from persistent device buffer
    if (is_full_layer[idx]) {
      bmrt_tensor_with_device(&in_tensors[2], dev_mask_full,
                              net->input_dtypes[2],
                              net->stages[0].input_shapes[2]);
    } else {
      bmrt_tensor_with_device(&in_tensors[2], dev_mask_sliding,
                              net->input_dtypes[2],
                              net->stages[0].input_shapes[2]);
    }

    // input[3]: input_ids - from persistent dev_buffer_ids
    bmrt_tensor_with_device(&in_tensors[3], dev_buffer_ids,
                            net->input_dtypes[3],
                            net->stages[0].input_shapes[3]);

    // input[4]: inputs_embeds - from dev_buffer (embedding output)
    bmrt_tensor_with_device(&in_tensors[4], dev_buffer,
                            net->input_dtypes[4],
                            net->stages[0].input_shapes[4]);

    if (is_shared_layer[idx]) {
      // input[5]: shared_k - source layer's past_key
      int source = kv_shared_source[idx];
      bmrt_tensor_with_device(&in_tensors[5], past_key[source],
                              net->input_dtypes[5],
                              net->stages[0].input_shapes[5]);
      // input[6]: shared_v - source layer's past_value
      bmrt_tensor_with_device(&in_tensors[6], past_value[source],
                              net->input_dtypes[6],
                              net->stages[0].input_shapes[6]);
    }

    // output[0]: output_states
    bmrt_tensor_with_device(&out_tensors[0],
                            net->stages[0].output_mems[0],
                            net->output_dtypes[0],
                            net->stages[0].output_shapes[0]);

    if (!is_shared_layer[idx]) {
      // output[1]: k_cache, output[2]: v_cache
      bmrt_tensor_with_device(&out_tensors[1],
                              net->stages[0].output_mems[1],
                              net->output_dtypes[1],
                              net->stages[0].output_shapes[1]);
      bmrt_tensor_with_device(&out_tensors[2],
                              net->stages[0].output_mems[2],
                              net->output_dtypes[2],
                              net->stages[0].output_shapes[2]);
    }

    // Dynamic shapes: reduce sequence dimension to token_length
    if (is_dynamic) {
      // input[0]: input_states [1, MAX_INPUT_LENGTH, H] -> [1, token_length, H]
      in_tensors[0].shape.dims[1] = token_length;
      // input[1]: position_ids [1, MAX_INPUT_LENGTH] -> [1, token_length]
      in_tensors[1].shape.dims[1] = token_length;
      // input[2]: attention_mask [1, 1, MAX_INPUT_LENGTH, MAX_INPUT_LENGTH] -> [1, 1, token_length, token_length]
      in_tensors[2].shape.dims[2] = token_length;
      in_tensors[2].shape.dims[3] = token_length;
      // input[3]: input_ids [1, MAX_INPUT_LENGTH] -> [1, token_length]
      in_tensors[3].shape.dims[1] = token_length;
      // input[4]: inputs_embeds [1, MAX_INPUT_LENGTH, H] -> [1, token_length, H]
      in_tensors[4].shape.dims[1] = token_length;
      if (is_shared_layer[idx]) {
        // input[5]: shared_k [1, MAX_INPUT_LENGTH, kv_heads, head_dim] -> [1, token_length, ...]
        in_tensors[5].shape.dims[1] = token_length;
        // input[6]: shared_v [1, MAX_INPUT_LENGTH, kv_heads, head_dim] -> [1, token_length, ...]
        in_tensors[6].shape.dims[1] = token_length;
      }
    }

    auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                     in_tensors.size(), out_tensors.data(),
                                     out_tensors.size(), true, false);
    assert(ret);

    if (is_dynamic) {
      d2d(hidden_state, net->stages[0].output_mems[0], 0, valid_bytes);
    } else {
      d2d(hidden_state, net->stages[0].output_mems[0]);
    }

    if (!is_shared_layer[idx]) {
      // Save k/v to past_key/past_value
      d2d(past_key[idx], net->stages[0].output_mems[1], 0,
          token_length * kv_bytes_per_token[idx]);
      d2d(past_value[idx], net->stages[0].output_mems[2], 0,
          token_length * kv_bytes_per_token[idx]);
    }
  }

  // Forward lm_head
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, hidden_state,
                     (token_length - 1) * hidden_bytes, hidden_bytes);
  net_launch(net_lm);

  int token = 0;
  if (lmhead_with_topk) {
    bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  } else if (generation_mode == "greedy") {
    token = greedy_search(lm_out_mem);
  } else if (generation_mode == "sample") {
    token = penalty_sample(lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

int Gemma4::forward_next() {
  int cur_token = visited_tokens[token_length - 1];

  // Full attention mask: visible for 0..token_length-2, masked for padding
  std::vector<uint16_t> full_attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    full_attention_mask[i] = mask_value;
  }
  // Sliding window mask: additionally mask positions outside window
  std::vector<uint16_t> sliding_attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    sliding_attention_mask[i] = mask_value;
  }
  int window = SLIDING_WINDOW;
  if (token_length > window) {
    for (int i = 0; i < token_length - window; i++) {
      sliding_attention_mask[i] = mask_value;
    }
  }

  // Shared layer variants: mask SEQLEN position and unmask cur_token position
  std::vector<uint16_t> full_attention_mask_shared = full_attention_mask;
  full_attention_mask_shared[SEQLEN] = mask_value;
  full_attention_mask_shared[token_length - 1] = 0;
  std::vector<uint16_t> sliding_attention_mask_shared = sliding_attention_mask;
  sliding_attention_mask_shared[SEQLEN] = mask_value;
  sliding_attention_mask_shared[token_length - 1] = 0;

  int32_t position_id = token_length - 1;

  // Write reusable inputs to device once
  bm_memcpy_s2d(bm_handle, dev_position_ids_cache, (void *)&position_id);
  bm_memcpy_s2d(bm_handle, dev_mask_full_cache, (void *)full_attention_mask.data());
  bm_memcpy_s2d(bm_handle, dev_mask_full_shared_cache, (void *)full_attention_mask_shared.data());
  bm_memcpy_s2d(bm_handle, dev_mask_sliding_cache, (void *)sliding_attention_mask.data());
  bm_memcpy_s2d(bm_handle, dev_mask_sliding_shared_cache, (void *)sliding_attention_mask_shared.data());

  // Embedding
  auto in_mem = net_embed_cache->stages[0].input_mems[0];
  auto out_mem = net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(net_embed_cache);
  d2d(dev_buffer_ids_cache, in_mem);
  d2d(dev_buffer_cache, out_mem);

  // Forward blocks
  auto hidden_state = net_blocks_cache[0]->stages[0].output_mems[0];
  d2d(hidden_state, dev_buffer_cache);
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &net = net_blocks_cache[idx];
    int input_num = net->input_num;
    int output_num = net->output_num;

    std::vector<bm_tensor_t> in_tensors(input_num);
    std::vector<bm_tensor_t> out_tensors(output_num);

    // input[0]: input_states (from previous layer or embedding)
    bmrt_tensor_with_device(&in_tensors[0], hidden_state, net->input_dtypes[0],
                            net->stages[0].input_shapes[0]);

    // input[1]: position_ids - from persistent device buffer
    bmrt_tensor_with_device(&in_tensors[1], dev_position_ids_cache,
                            net->input_dtypes[1],
                            net->stages[0].input_shapes[1]);

    // input[2]: attention_mask - from persistent device buffer
    bm_device_mem_t &cur_mask_mem =
        (is_full_layer[idx] && !is_shared_layer[idx]) ? dev_mask_full_cache :
        (is_full_layer[idx] && is_shared_layer[idx]) ? dev_mask_full_shared_cache :
        (!is_full_layer[idx] && !is_shared_layer[idx]) ? dev_mask_sliding_cache :
        dev_mask_sliding_shared_cache;
    bmrt_tensor_with_device(&in_tensors[2], cur_mask_mem,
                            net->input_dtypes[2],
                            net->stages[0].input_shapes[2]);

    // input[3]: input_ids - from persistent dev_buffer_ids_cache
    bmrt_tensor_with_device(&in_tensors[3], dev_buffer_ids_cache,
                            net->input_dtypes[3],
                            net->stages[0].input_shapes[3]);

    // input[4]: inputs_embeds - reference net_embed_cache's output
    bmrt_tensor_with_device(&in_tensors[4], dev_buffer_cache,
                            net->input_dtypes[4],
                            net->stages[0].input_shapes[4]);

    if (!is_shared_layer[idx]) {
      // Normal layer: input[5]=history_k, input[6]=history_v
      bmrt_tensor_with_device(&in_tensors[5], past_key[idx],
                              net->input_dtypes[5],
                              net->stages[0].input_shapes[5]);
      bmrt_tensor_with_device(&in_tensors[6], past_value[idx],
                              net->input_dtypes[6],
                              net->stages[0].input_shapes[6]);
    } else {
      // Shared layer: input[5]=shared_k, input[6]=shared_v from source layer
      int source = kv_shared_source[idx];
      bmrt_tensor_with_device(&in_tensors[5], past_key[source],
                              net->input_dtypes[5],
                              net->stages[0].input_shapes[5]);
      bmrt_tensor_with_device(&in_tensors[6], past_value[source],
                              net->input_dtypes[6],
                              net->stages[0].input_shapes[6]);
    }

    // output[0]: output_states
    bmrt_tensor_with_device(&out_tensors[0],
                            net->stages[0].output_mems[0],
                            net->output_dtypes[0],
                            net->stages[0].output_shapes[0]);

    if (!is_shared_layer[idx]) {
      // Normal layer: output[1]=k_cache, output[2]=v_cache
      // Written at offset (token_length-1)*kv_bytes_per_token in past_key/past_value
      int kv_offset_bytes = (token_length - 1) * kv_bytes_per_token[idx];
      auto k_mem = bm_mem_from_device(
          past_key[idx].u.device.device_addr + kv_offset_bytes,
          kv_bytes_per_token[idx]);
      auto v_mem = bm_mem_from_device(
          past_value[idx].u.device.device_addr + kv_offset_bytes,
          kv_bytes_per_token[idx]);
      bmrt_tensor_with_device(&out_tensors[1], k_mem, net->output_dtypes[1],
                              net->stages[0].output_shapes[1]);
      bmrt_tensor_with_device(&out_tensors[2], v_mem, net->output_dtypes[2],
                              net->stages[0].output_shapes[2]);
    }

    auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                     in_tensors.size(), out_tensors.data(),
                                     out_tensors.size(), true, false);
    assert(ret);
    d2d(hidden_state, net->stages[0].output_mems[0]);
  }

  // Forward lm_head
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  d2d(lm_in_mem, hidden_state);
  net_launch(net_lm);

  int token = 0;
  if (lmhead_with_topk) {
    bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  } else if (generation_mode == "greedy") {
    token = greedy_search(lm_out_mem);
  } else if (generation_mode == "sample") {
    token = penalty_sample(lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Gemma4>(m, "Gemma4")
      .def(pybind11::init<>())
      .def("init", &Gemma4::init)
      .def("forward_embed", &Gemma4::forward_embed)
      .def("forward_vit", &Gemma4::forward_vit, py::arg("pixel_values"),
           py::arg("pixel_position_ids"), py::arg("vit_offsets"),
           py::arg("mm_tokens_actual"), py::arg("is_video") = false)
      .def("forward_audio", &Gemma4::forward_audio, py::arg("audio_features"),
           py::arg("audio_offset"))
      .def("forward_first", &Gemma4::forward_first)
      .def("forward_next", &Gemma4::forward_next)
      .def("deinit", &Gemma4::deinit)
      .def_readwrite("SEQLEN", &Gemma4::SEQLEN)
      .def_readwrite("MAX_INPUT_LENGTH", &Gemma4::MAX_INPUT_LENGTH)
      .def_readwrite("token_length", &Gemma4::token_length)
      .def_readwrite("generation_mode", &Gemma4::generation_mode)
      .def_readwrite("penalty", &Gemma4::penalty)
      .def_readwrite("temperature", &Gemma4::temperature)
      .def_readwrite("top_k", &Gemma4::top_k)
      .def_readwrite("top_p", &Gemma4::top_p)
      .def_readwrite("SLIDING_WINDOW", &Gemma4::SLIDING_WINDOW)
      .def_readwrite("ID_IMAGE_PAD", &Gemma4::ID_IMAGE_PAD)
      .def_readwrite("ID_VIDEO_PAD", &Gemma4::ID_VIDEO_PAD)
      .def_readwrite("ID_AUDIO_PAD", &Gemma4::ID_AUDIO_PAD)
      .def_readwrite("ID_BOA", &Gemma4::ID_BOA)
      .def_readwrite("ID_EOA", &Gemma4::ID_EOA)
      .def_readonly("AUDIO_MEL", &Gemma4::AUDIO_MEL)
      .def_readonly("AUDIO_TOKENS", &Gemma4::AUDIO_TOKENS);
}
