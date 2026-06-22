//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "embed_vit.hpp"
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
// EmbedVit
//===------------------------------------------------------------===//

void EmbedVit::init_by_names() {

  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  vit_dynamic = net_vit->is_dynamic;

  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_embed->stages[0].output_shapes[0].dims[2];
  SEQLEN = MAX_INPUT_LENGTH;
  MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];
  MAX_PIXELS = MAX_PATCHES * 16 * 16;
  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  printf("Max Pixels: %d*%d*%d\n", MAX_PATCHES / 4, 32, 32);
}

void EmbedVit::init(int dev_id, std::string model_path) {
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

  vit_run = false;
}

void EmbedVit::deinit() {
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void EmbedVit::allocate_history_buffer(int seq_len) {
  support_history = true;
  if (seq_len <= MAX_INPUT_LENGTH) {
    return; // existing buffer already large enough
  }
  SEQLEN = seq_len;
}

void EmbedVit::forward_embed(ArrayInt const &tokens) {
  token_length = tokens.size();
  sys_buffer.assign(token_length * HIDDEN_SIZE, 0);
  if (!support_history) {
    assert(token_length <= MAX_INPUT_LENGTH);
  }
  // chunked embed: each launch handles up to MAX_INPUT_LENGTH tokens, and
  // the resulting hidden states are appended into sys_buffer with offset.
  // For inputs that fit in one chunk this loop runs a single iteration.
  for (int i = 0; i < token_length; i += MAX_INPUT_LENGTH) {
    std::vector<bm_tensor_t> in_tensors;
    std::vector<bm_tensor_t> out_tensors;
    init_tensors(net_embed, in_tensors, out_tensors);
    int real_len = std::min(MAX_INPUT_LENGTH, token_length - i);
    if (real_len != MAX_INPUT_LENGTH) {
      empty(bm_handle, in_tensors[0].device_mem);
    }
    bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                          (void *)(tokens.data() + i), real_len * sizeof(int));
    net_launch(p_bmrt, net_embed, in_tensors, out_tensors);
    int offset = i * HIDDEN_SIZE;
    bm_thread_sync(bm_handle);
    bm_memcpy_d2s_partial(bm_handle, sys_buffer.data() + offset,
                          out_tensors[0].device_mem,
                          real_len * HIDDEN_SIZE * sizeof(uint16_t));
  }
}

void EmbedVit::forward_vit(const float *pixel_values,
                           ArrayInt const &position_ids,
                           ArrayInt const &pos_idx,
                           ArrayFloat const &pos_weight,
                           ArrayInt const &grid_thw, int vit_offset) {
  const int *p_thw = grid_thw.data();
  int t = p_thw[0];
  int h = p_thw[1];
  int w = p_thw[2];
  int hw = t * h * w;
  int num_pixels = hw * VIT_DIMS;
  assert((int)position_ids.size() == (hw * 2));
  assert((int)pos_idx.size() == 4 * hw);
  assert((int)pos_weight.size() == 4 * hw);
  auto p_position_ids = position_ids.data();
  auto p_pos_idx = pos_idx.data();
  auto p_pos_weight = pos_weight.data();
  empty_net(bm_handle, net_vit);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_vit, in_tensors, out_tensors);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)pixel_values, num_pixels * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                        (void *)p_position_ids,
                        position_ids.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem, (void *)p_pos_idx,
                        pos_idx.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[3].device_mem,
                        (void *)p_pos_weight,
                        pos_weight.size() * sizeof(float));
  if (vit_dynamic) {
    in_tensors[0].shape.dims[0] = hw;
    in_tensors[1].shape.dims[0] = hw;
    in_tensors[2].shape.dims[0] = hw;
    in_tensors[3].shape.dims[0] = hw;
  } else {
    std::vector<float> attention_mask(MAX_PATCHES * MAX_PATCHES, -10000.0f);
    for (int i = 0; i < hw; i++) {
      auto row_begin = attention_mask.begin() + i * MAX_PATCHES;
      std::fill(row_begin, row_begin + hw, 0.0f);
    }
    bm_memcpy_s2d(bm_handle, in_tensors[4].device_mem,
                  (void *)attention_mask.data());
  }

  net_launch(p_bmrt, net_vit, in_tensors, out_tensors);

  // concatenate text embedding and image embedding
  bm_thread_sync(bm_handle);
  int dst_offset = vit_offset * HIDDEN_SIZE;
  int vit_size = hw / 4 * HIDDEN_SIZE;
  bm_memcpy_d2s_partial(bm_handle, sys_buffer.data() + dst_offset,
                        out_tensors[0].device_mem, vit_size * sizeof(uint16_t));
  vit_run = true;
}

ArrayUint16 EmbedVit::forward_embed_cache(int token) {
  assert(net_embed_cache != nullptr &&
         "embedding_cache network not found in embed_vit.bmodel");
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed_cache, in_tensors, out_tensors);
  bm_memcpy_s2d(bm_handle, in_tensors[0].device_mem, (void *)&token);
  net_launch(p_bmrt, net_embed_cache, in_tensors, out_tensors);
  ArrayUint16 result(HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, result.data(), out_tensors[0].device_mem,
                        HIDDEN_SIZE * sizeof(uint16_t));
  return result;
}

ArrayUint16 &EmbedVit::get_hidden_states() { return sys_buffer; }
