//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "chat.hpp"

//===------------------------------------------------------------===//
// Empty Func
//===------------------------------------------------------------===//
void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
}

void empty_in_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
                  int stage_idx = 0) {
  for (int i = 0; i < net->input_num; i++) {
    empty(bm_handle, net->stages[stage_idx].input_mems[i]);
  }
}

void empty_out_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
                   int stage_idx = 0) {
  for (int i = 0; i < net->output_num; i++) {
    empty(bm_handle, net->stages[stage_idx].output_mems[i]);
  }
}

void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
               int stage_idx = 0) {
  empty_in_net(bm_handle, net, stage_idx);
  empty_out_net(bm_handle, net, stage_idx);
}

void Qwen2_5VL::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void Qwen2_5VL::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Qwen2_5VL::init(int dev_id, std::string model_path) {

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
  printf("Done!\n");

  // net embed and lm_head
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];
  MAX_PIXELS = MAX_PATCHES * 14 * 14;
  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 4) / 2;
  printf("Num Layers:%d\n", NUM_LAYERS);
  printf("Max Pixels: %d*%d*%d\n", MAX_PATCHES / 4, 28, 28);

  if (net_embed_cache->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // ATTENTION_MASK in fp16
  } else if (net_embed_cache->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // ATTENTION_MASK in bfloat16
  } else {
    std::cerr << "\nError: Unsupported Dtype: "
              << net_embed_cache->output_dtypes[0];
    throw std::runtime_error("Invalid attention dtype");
  }

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
  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);
}

void Qwen2_5VL::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void Qwen2_5VL::forward_embed(ArrayInt const &tokens) {
  std::vector<int> input_ids(SEQLEN, 0);
  auto num = tokens.size();
  const int *p_tokens = tokens.data();
  std::copy(p_tokens, p_tokens + num, input_ids.data());

  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)input_ids.data());
  net_launch(net_embed);
  d2d(dev_buffer, out_mem);
  token_length = tokens.size();
}

void Qwen2_5VL::forward_vit(ArrayFloat const &pixel_values,
                            ArrayInt const &position_ids,
                            ArrayFloat const &full_attn_mask,
                            ArrayFloat const &window_attn_mask,
                            ArrayInt const &grid_thw,
                            ArrayInt const &reverse_indices, int vit_offset) {
  // 修正拼写错误
  const int *p_thw = grid_thw.data();
  int t = p_thw[0];
  int h = p_thw[1];
  int w = p_thw[2];
  int hw = t * h * w;

  assert(full_attn_mask.size() == static_cast<size_t>(hw * hw));
  assert(window_attn_mask.size() == static_cast<size_t>(hw * hw));
  assert(pixel_values.size() == static_cast<size_t>(hw * VIT_DIMS));
  assert(position_ids.size() == static_cast<size_t>(hw * 2));
  assert(reverse_indices.size() == static_cast<size_t>(hw / 4));
  const float *p_pixel_values = pixel_values.data();
  const int *p_position_ids = position_ids.data();
  const float *p_full = full_attn_mask.data();
  const float *p_window = window_attn_mask.data();
  const int *p_reverse_indices = reverse_indices.data();

  empty_net(bm_handle, net_vit);
  auto &vit_in0_mem = net_vit->stages[0].input_mems[0];
  auto &vit_in1_mem = net_vit->stages[0].input_mems[1];
  auto &vit_in2_mem = net_vit->stages[0].input_mems[2];
  auto &vit_in3_mem = net_vit->stages[0].input_mems[3];
  auto &vit_in4_mem = net_vit->stages[0].input_mems[4];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  bm_memcpy_s2d_partial(bm_handle, vit_in0_mem, (void *)p_pixel_values,
                        pixel_values.size() * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, vit_in1_mem, (void *)p_position_ids,
                        position_ids.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, vit_in4_mem, (void *)p_reverse_indices,
                        reverse_indices.size() * sizeof(int));
  if (full_attn_mask.size() == MAX_PATCHES * MAX_PATCHES) {
    bm_memcpy_s2d(bm_handle, vit_in2_mem, (void *)p_full);
    bm_memcpy_s2d(bm_handle, vit_in3_mem, (void *)p_window);
  } else {
    std::vector<float> mask_full(MAX_PATCHES * MAX_PATCHES, -10000.0f);
    std::vector<float> mask_window(MAX_PATCHES * MAX_PATCHES, -10000.0f);

    for (int i = 0; i < hw; i++) {
      int mask_offset = i * MAX_PATCHES;
      int ori_offset = i * hw;
      std::copy(p_full + ori_offset, p_full + ori_offset + hw,
                mask_full.begin() + mask_offset);
      std::copy(p_window + ori_offset, p_window + ori_offset + hw,
                mask_window.begin() + mask_offset);
    }
    bm_memcpy_s2d(bm_handle, vit_in2_mem, (void *)mask_full.data());
    bm_memcpy_s2d(bm_handle, vit_in3_mem, (void *)mask_window.data());
  }
  // launch vit
  net_launch(net_vit);

  // concatenante texting embedding and image embedding
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = hw / 4 * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset, vit_out_mem, 0,
                     vit_size);
}

void Qwen2_5VL::head_launch(const bm_net_info_t *net,
                            bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device(&in_tensors[0], logits_mem, net->input_dtypes[0],
                          net->stages[0].input_shapes[0]);

  for (int i = 1; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[0].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[0].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net->stages[0].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[0].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

int Qwen2_5VL::forward_first(ArrayInt const &position_ids) {

  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < token_length; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  const int *p_ids = position_ids.data();

  std::vector<int> position_ids_pad(3 * SEQLEN, 0);
  int ori_length = position_ids.size() / 3;
  for (int i = 0; i < 3; i++) {
    int ori_offset = i * ori_length;
    int dst_offset = i * SEQLEN;
    std::copy(p_ids + ori_offset, p_ids + ori_offset + ori_length,
              position_ids_pad.begin() + dst_offset);
  }

  auto out_mem = dev_buffer;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_ids_pad.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
  }

  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * bytes, bytes);
  net_launch(net_lm);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  token_length++;
  return token;
}

int Qwen2_5VL::forward_next(ArrayInt const &position_ids) {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  assert(position_ids.size() == 3);
  const int *p_ids = position_ids.data();
  // embedding
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];

  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  d2d(in_mem, lm_out_mem);
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
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)p_ids);
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
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  token_length++;
  return token;
}