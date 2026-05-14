//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "block.hpp"

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

void Block::net_launch_decode(int idx, int kv_offset, const int *pos_id,
                              std::vector<uint16_t> &attention_mask) {
  auto &net = net_blocks_cache[idx];
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net, in_tensors, out_tensors);
  if (idx == 0) {
    bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem, (void *)pos_id);
    bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                  (void *)attention_mask.data());
  } else {
    in_tensors[0].device_mem =
        net_blocks_cache[idx - 1]->stages[0].output_mems[0];
    in_tensors[1].device_mem = net_blocks_cache[0]->stages[0].input_mems[1];
    in_tensors[2].device_mem = net_blocks_cache[0]->stages[0].input_mems[2];
  }
  out_tensors[1].device_mem = bm_mem_from_device(
      past_key[idx].u.device.device_addr + kv_offset, KV_BYTES);
  out_tensors[2].device_mem = bm_mem_from_device(
      past_value[idx].u.device.device_addr + kv_offset, KV_BYTES);

  net_launch(p_bmrt, net, in_tensors, out_tensors);
}

void Block::clear_history() {
  if (!support_history) {
    return;
  }
  for (int i = 0; i < num_blocks; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
}

void Block::init_by_names() {
  auto num_nets = bmrt_get_network_number(p_bmrt);
  const char **net_names = nullptr;
  bmrt_get_network_names(p_bmrt, &net_names);

  net_add = nullptr;

  // Parse network names to find block index range automatically
  int min_idx = 10000, max_idx = 0;
  const std::string block_cache_prefix = "block_cache_";
  for (int i = 0; i < num_nets; i++) {
    std::string name(net_names[i]);
    if (name == "add") {
      net_add = bmrt_get_network_info(p_bmrt, "add");
      continue;
    }
    if (name.compare(0, block_cache_prefix.size(), block_cache_prefix) == 0) {
      int idx = std::stoi(name.substr(block_cache_prefix.size()));
      min_idx = std::min(min_idx, idx);
      max_idx = std::max(max_idx, idx);
    }
  }
  free(net_names);

  start_idx = min_idx;
  num_blocks = max_idx - min_idx + 1;

  for (int i = start_idx; i <= max_idx; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }
  if (net_blocks[0]->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // float16
  } else if (net_blocks[0]->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // -9984 by bfloat16
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }
  support_history = net_blocks[0]->input_num == 5;
  is_dynamic = net_blocks[0]->is_dynamic;
  history_length = 0;
  MAX_INPUT_LENGTH = net_blocks[0]->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_blocks_cache[0]->stages[0].input_shapes[0].dims[2];
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  KV_BYTES =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  PREFILL_KV_LENGTH = 0;
  if (support_history) {
    PREFILL_KV_LENGTH = net_blocks[0]->stages[0].input_shapes[3].dims[1];
    printf("History Support: True\n");
  } else {
    printf("History Support: False\n");
  }
}

void Block::init(int dev_id, std::string model_path) {
  std::cout << "Device [ " << dev_id << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  assert(BM_SUCCESS == status);

  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  bm_thread_sync(bm_handle);
  printf("Done!\n");
  print_devmem_info(bm_handle);

  init_by_names();

  visited_tokens.resize(SEQLEN);

  past_key.resize(num_blocks);
  past_value.resize(num_blocks);
  for (int i = 0; i < num_blocks; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
}

void Block::deinit() {
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

ArrayUint16 Block::forward_first(ArrayInt const &position_ids,
                                 ArrayUint16 &hidden_states,
                                 ArrayUint162D &deepstacks) {
  if (support_history) {
    return forward_first_with_kv(position_ids, hidden_states, deepstacks);
  }
  const int *p_ids = position_ids.data();
  std::vector<int> position_ids_pad;
  std::vector<uint16_t> attention_mask;
  if (is_dynamic) {
    attention_mask.assign(token_length * token_length, mask_value);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * token_length + j] = 0;
      }
    }
    position_ids_pad.assign(3 * token_length, 0);
    assert((int)position_ids.size() == token_length * 3);
    std::copy(p_ids, p_ids + token_length * 3, position_ids_pad.begin());
  } else {
    int length = MAX_INPUT_LENGTH;
    attention_mask.assign(length * length, mask_value);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * length + j] = 0;
      }
    }
    position_ids_pad.assign(3 * length, 0);
    int ori_length = position_ids.size() / 3;
    for (int i = 0; i < 3; i++) {
      int ori_offset = i * ori_length;
      int dst_offset = i * length;
      std::copy(p_ids + ori_offset, p_ids + ori_offset + ori_length,
                position_ids_pad.begin() + dst_offset);
    }
  }

  bm_device_mem_t out_mem;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  for (int idx = 0; idx < num_blocks; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors);
    if (idx == 0) {
      bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                            hidden_states.data(),
                            hidden_states.size() * sizeof(uint16_t));
    } else {
      d2d(bm_handle, in_tensors[0].device_mem, out_mem);
    }
    if (is_dynamic) {
      if (idx == 0) {
        bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                              (void *)position_ids_pad.data(),
                              token_length * 3 * sizeof(int));
        bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                              (void *)attention_mask.data(),
                              token_length * token_length * sizeof(uint16_t));
      }
      in_tensors[0].shape.dims[1] = token_length;
      in_tensors[1].shape.dims[1] = token_length;
      in_tensors[2].shape.dims[2] = token_length;
      in_tensors[2].shape.dims[3] = token_length;
    } else {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                      (void *)position_ids_pad.data());
        bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                      (void *)attention_mask.data());
      }
    }
    net_launch(p_bmrt, net_blocks[idx], in_tensors, out_tensors);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    if (idx < (int)deepstacks.size()) {
      init_tensors(net_add, in_tensors, out_tensors);
      in_tensors[0].device_mem = out_mem;
      bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                            (void *)deepstacks[idx].data(),
                            deepstacks[idx].size() * sizeof(uint16_t));
      net_launch(p_bmrt, net_add, in_tensors, out_tensors);
      out_mem = net_add->stages[0].output_mems[0];
    }
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], 0,
                       net_blocks[idx]->stages[0].output_mems[1], 0,
                       KV_BYTES * token_length);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], 0,
                       net_blocks[idx]->stages[0].output_mems[2], 0,
                       KV_BYTES * token_length);
  }
  bm_thread_sync(bm_handle);
  int bytes = token_length * HIDDEN_SIZE * sizeof(uint16_t);
  ArrayUint16 result(token_length * HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, result.data(), out_mem, bytes);
  return result;
}

ArrayUint16 Block::forward_first_with_kv(ArrayInt const &position_ids,
                                         ArrayUint16 &hidden_states,
                                         ArrayUint162D &deepstacks) {
  int max_kv_length = MAX_INPUT_LENGTH + PREFILL_KV_LENGTH;
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * max_kv_length,
                                       mask_value);
  auto old_length = history_length;
  history_length += token_length;
  assert(history_length < SEQLEN);
  assert(old_length <= PREFILL_KV_LENGTH);
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < old_length; j++) {
      attention_mask[i * max_kv_length + j] = 0;
    }
    for (int j = 0; j <= i; j++) {
      attention_mask[i * max_kv_length + j + PREFILL_KV_LENGTH] = 0;
    }
  }

  const int *p_ids = position_ids.data();

  std::vector<int> position_ids_pad(3 * MAX_INPUT_LENGTH, 0);
  int ori_length = position_ids.size() / 3;
  assert(ori_length == token_length);
  assert(ori_length <= MAX_INPUT_LENGTH);
  for (int i = 0; i < 3; i++) {
    int ori_offset = i * ori_length;
    int dst_offset = i * MAX_INPUT_LENGTH;
    std::copy(p_ids + ori_offset, p_ids + ori_offset + ori_length,
              position_ids_pad.begin() + dst_offset);
  }
  bm_device_mem_t out_mem;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  for (int idx = 0; idx < num_blocks; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors);
    if (idx == 0) {
      bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                            hidden_states.data(),
                            hidden_states.size() * sizeof(uint16_t));
    } else {
      d2d(bm_handle, in_tensors[0].device_mem, out_mem);
    }
    if (old_length > 0) {
      d2d(bm_handle, in_tensors[3].device_mem, past_key[idx], 0,
          KV_BYTES * old_length);
      d2d(bm_handle, in_tensors[4].device_mem, past_value[idx], 0,
          KV_BYTES * old_length);
    } else if (idx == 0) {
      empty(bm_handle, in_tensors[3].device_mem);
      empty(bm_handle, in_tensors[4].device_mem);
    }
    bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                  (void *)position_ids_pad.data());
    bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                  (void *)attention_mask.data());
    net_launch(p_bmrt, net_blocks[idx], in_tensors, out_tensors);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    if (idx < (int)deepstacks.size()) {
      init_tensors(net_add, in_tensors, out_tensors);
      in_tensors[0].device_mem = out_mem;
      bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                            (void *)deepstacks[idx].data(),
                            deepstacks[idx].size() * sizeof(uint16_t));
      net_launch(p_bmrt, net_add, in_tensors, out_tensors);
      out_mem = net_add->stages[0].output_mems[0];
    }
    auto &out1_mem = net_blocks[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks[idx]->stages[0].output_mems[2];
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], old_length * KV_BYTES,
                       out1_mem, 0, KV_BYTES * token_length);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], old_length * KV_BYTES,
                       out2_mem, 0, KV_BYTES * token_length);
  }

  bm_thread_sync(bm_handle);
  int bytes = token_length * HIDDEN_SIZE * sizeof(uint16_t);
  ArrayUint16 result(token_length * HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, result.data(), out_mem, bytes);
  return result;
}

ArrayUint16 Block::forward_next(ArrayInt const &position_ids,
                                ArrayUint16 &hidden_states) {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = history_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  assert(position_ids.size() == 3);
  const int *p_ids = position_ids.data();
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;

  // blocks
  int kv_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (history_length - 1) * kv_bytes;
  for (int idx = 0; idx < num_blocks; idx++) {
    if (idx == 0) {
      bm_memcpy_s2d_partial(
          bm_handle, net_blocks_cache[idx]->stages[0].input_mems[0],
          hidden_states.data(), hidden_states.size() * sizeof(uint16_t));
    }
    net_launch_decode(idx, token_offset, p_ids, attention_mask);
  }
  auto out_mem = net_blocks_cache[num_blocks - 1]->stages[0].output_mems[0];
  bm_thread_sync(bm_handle);
  int out_bytes = HIDDEN_SIZE * sizeof(uint16_t);
  ArrayUint16 results(HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, results.data(), out_mem, out_bytes);
  return results;
}
