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

void Block::net_launch_decode(int local_idx, int kv_offset, const int *pos_id,
                              std::vector<uint16_t> &attention_mask,
                              int stage_idx) {
  auto &net = net_blocks_cache[local_idx];
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net, in_tensors, out_tensors, stage_idx);
  bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem, (void *)pos_id);
  bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                (void *)attention_mask.data());
  out_tensors[1].device_mem = bm_mem_from_device(
      past_key[local_idx].u.device.device_addr + kv_offset, KV_BYTES);
  out_tensors[2].device_mem = bm_mem_from_device(
      past_value[local_idx].u.device.device_addr + kv_offset, KV_BYTES);

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

  auto is_exist = [net_names, num_nets](const std::string &name) {
    for (int i = 0; i < num_nets; i++) {
      if (name == net_names[i]) {
        return true;
      }
    }
    return false;
  };

  // Discover the global block index range from network names "block_cache_<i>"
  int min_idx = 100000, max_idx = -1;
  const std::string block_cache_prefix = "block_cache_";
  for (int i = 0; i < num_nets; i++) {
    std::string name(net_names[i]);
    if (name.compare(0, block_cache_prefix.size(), block_cache_prefix) == 0) {
      int idx = std::stoi(name.substr(block_cache_prefix.size()));
      min_idx = std::min(min_idx, idx);
      max_idx = std::max(max_idx, idx);
    }
  }
  if (max_idx < 0) {
    free(net_names);
    throw std::runtime_error("No block_cache_* networks found in bmodel");
  }

  start_idx = min_idx;
  num_blocks = max_idx - min_idx + 1;

  for (int i = start_idx; i <= max_idx; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    auto cache_net = bmrt_get_network_info(p_bmrt, cache_name.c_str());
    net_blocks_cache.emplace_back(cache_net);
    if (is_FA(i)) {
      auto decode_stage_num = cache_net->stage_num;
      if (decode_stage_len.empty()) {
        for (int j = 0; j < decode_stage_num; j++) {
          decode_stage_len.push_back(
              cache_net->stages[j].input_shapes[3].dims[1]);
        }
      } else {
        assert(decode_stage_num == (int)decode_stage_len.size());
        for (int j = 0; j < decode_stage_num; j++) {
          assert(cache_net->stages[j].input_shapes[3].dims[1] ==
                 decode_stage_len[j]);
        }
      }
    }
  }

  if (net_blocks[0]->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // float16
  } else if (net_blocks[0]->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // -9984 by bfloat16
  } else {
    free(net_names);
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }

  // Find the first FA layer in this Block instance to introspect shapes.
  int first_fa_local = -1;
  for (int i = 0; i < num_blocks; i++) {
    if (is_FA(start_idx + i)) {
      first_fa_local = i;
      break;
    }
  }
  if (first_fa_local < 0) {
    free(net_names);
    throw std::runtime_error(
        "Block instance must contain at least one Full-Attention layer "
        "(every " +
        std::to_string(FA_INTERVAL) + "-th layer)");
  }

  // support_history is enabled by the bmodel when block_kv_<first_fa_global>
  // exists. The kv net is the full-attention prefill that concatenates history
  // K/V; the fresh first-time prefill uses block_ (no history input).
  std::string kv_name =
      "block_kv_" + std::to_string(start_idx + first_fa_local);
  support_history = is_exist(kv_name);

  // Load kv nets for FA layers (nullptr for non-FA layers).
  net_blocks_kv.assign(num_blocks, nullptr);
  if (support_history) {
    for (int i = 0; i < num_blocks; i++) {
      int global_idx = start_idx + i;
      if (!is_FA(global_idx)) {
        continue;
      }
      auto name = "block_kv_" + std::to_string(global_idx);
      if (is_exist(name)) {
        net_blocks_kv[i] = bmrt_get_network_info(p_bmrt, name.c_str());
      }
    }
  }

  auto fa_block = net_blocks[first_fa_local];
  auto fa_cache = net_blocks_cache[first_fa_local];
  prefill_mask = fa_block->input_num == (support_history ? 5 : 3);
  history_length = 0;
  MAX_INPUT_LENGTH = fa_block->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = fa_cache->stages[0].input_shapes[0].dims[2];
  SEQLEN = fa_cache->stages[0].input_shapes[3].dims[1];
  KV_BYTES = bm_mem_get_device_size(fa_cache->stages[0].output_mems[1]);
  PREFILL_KV_LENGTH = 0;
  if (support_history) {
    PREFILL_KV_LENGTH =
        net_blocks_kv[first_fa_local]->stages[0].input_shapes[3].dims[1];
    printf("History Support: True (block_kv detected)\n");
  } else {
    printf("History Support: False\n");
  }
  free(net_names);
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

  past_key.resize(num_blocks);
  past_value.resize(num_blocks);
  for (int i = 0; i < num_blocks; i++) {
    if (is_FA(start_idx + i)) {
      // Full-Attention layer: kv cache lives in the cache net's input mems 3/4.
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    } else {
      // Linear/recurrent layer: reuse input_mems[1]/[2] as conv/recurrent
      // state.
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[1];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[2];
    }
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  auto buffer_size =
      bm_mem_get_device_size(net_blocks[0]->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);
}

void Block::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

ArrayUint16 Block::forward_first(ArrayInt const &position_ids,
                                 ArrayUint16 &hidden_states) {
  if (support_history) {
    return forward_first_with_kv(position_ids, hidden_states);
  }
  const int *p_ids = position_ids.data();
  std::vector<int> position_ids_pad;
  std::vector<uint16_t> attention_mask;
  if (prefill_mask) {
    attention_mask.assign(token_length * token_length, mask_value);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * token_length + j] = 0;
      }
    }
  }
  position_ids_pad.assign(3 * token_length, 0);
  assert((int)position_ids.size() == token_length * 3);
  std::copy(p_ids, p_ids + token_length * 3, position_ids_pad.begin());

  bm_device_mem_t out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  for (int idx = 0; idx < num_blocks; idx++) {
    int global_idx = start_idx + idx;
    bool fa = is_FA(global_idx);
    init_tensors(net_blocks[idx], in_tensors, out_tensors);
    out_tensors[0].device_mem = out_mem;
    if (idx == 0) {
      bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                            hidden_states.data(),
                            hidden_states.size() * sizeof(uint16_t));
    } else {
      d2d(bm_handle, in_tensors[0].device_mem, out_mem);
    }

    if (fa) {
      bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                            (void *)position_ids_pad.data(),
                            token_length * 3 * sizeof(int));
      if (prefill_mask) {
        bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                              (void *)attention_mask.data(),
                              token_length * token_length * sizeof(uint16_t));
        in_tensors[2].shape.dims[2] = token_length;
        in_tensors[2].shape.dims[3] = token_length;
      }
      in_tensors[0].shape.dims[1] = token_length;
      in_tensors[1].shape.dims[1] = token_length;
    } else {
      // Non-FA layer: input[1] is the recurrent state (zeroed for prefill).
      in_tensors[0].shape.dims[1] = token_length;
      empty(bm_handle, in_tensors[1].device_mem);
    }

    net_launch(p_bmrt, net_blocks[idx], in_tensors, out_tensors);

    if (fa) {
      bm_memcpy_d2d_byte(bm_handle, past_key[idx], 0,
                         net_blocks[idx]->stages[0].output_mems[1], 0,
                         KV_BYTES * token_length);
      bm_memcpy_d2d_byte(bm_handle, past_value[idx], 0,
                         net_blocks[idx]->stages[0].output_mems[2], 0,
                         KV_BYTES * token_length);
    } else {
      // reuse key as conv state, value as recurrent state
      d2d(bm_handle, past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
      d2d(bm_handle, past_value[idx], net_blocks[idx]->stages[0].input_mems[1]);
    }
  }
  bm_thread_sync(bm_handle);
  int bytes = token_length * HIDDEN_SIZE * sizeof(uint16_t);
  ArrayUint16 result(token_length * HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, result.data(), out_mem, bytes);
  return result;
}

ArrayUint16 Block::forward_first_with_kv(ArrayInt const &position_ids,
                                         ArrayUint16 &hidden_states) {
  // Chunked prefill with KV cache, adapted to the pipeline-parallel layout:
  // hidden_states arrives on the host (transferred from the previous device),
  // so each chunk is S2D'd into the first layer's input mem, processed across
  // all layers with on-device D2D, and the chunk output is D2S'd back to host.
  assert(history_length + token_length < SEQLEN);
  assert(prefill_mask == false);
  assert((int)position_ids.size() == 3 * token_length);
  ArrayInt pos_ids(3 * MAX_INPUT_LENGTH, 0);

  bm_device_mem_t out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  int k_idx = 2;
  int old_kvlen = (history_length > 0) ? (history_length - 1) : 0;

  // Holds the full sequence output to pass to the next Block / LmHead.
  ArrayUint16 result(token_length * HIDDEN_SIZE);

  for (int t = 0; t < token_length; t += MAX_INPUT_LENGTH) {
    auto old_length = history_length;
    int cur_len = std::min(MAX_INPUT_LENGTH, token_length - t);
    history_length += cur_len;
    // copy position ids with offset (chunk t of each of the 3 dimensions)
    for (int i = 0; i < 3; i++) {
      std::copy(position_ids.data() + i * token_length + t,
                position_ids.data() + i * token_length + t + cur_len,
                pos_ids.data() + i * cur_len);
    }

    assert(old_length <= PREFILL_KV_LENGTH);
    for (int idx = 0; idx < num_blocks; idx++) {
      int global_idx = start_idx + idx;
      bool fa = is_FA(global_idx);
      // Fresh first-time prefill uses block_ (no history); subsequent chunks
      // concatenate history K/V via block_kv_.
      auto &net = (old_kvlen > 0 && fa) ? net_blocks_kv[idx] : net_blocks[idx];
      init_tensors(net, in_tensors, out_tensors);
      out_tensors[0].device_mem = out_mem;

      if (idx == 0) {
        // first block: copy chunk input from host hidden_states with offset
        bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                              (void *)(hidden_states.data() + t * HIDDEN_SIZE),
                              cur_len * HIDDEN_SIZE * sizeof(uint16_t));
      } else {
        d2d(bm_handle, in_tensors[0].device_mem, out_mem, 0,
            cur_len * HIDDEN_SIZE * sizeof(uint16_t));
      }
      if (fa) {
        bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                              (void *)(pos_ids.data()),
                              cur_len * 3 * sizeof(int));
        in_tensors[0].shape.dims[1] = cur_len;
        in_tensors[1].shape.dims[1] = cur_len;
        // copy old kv to new kv with offset
        if (old_kvlen > 0) {
          d2d(bm_handle, in_tensors[k_idx].device_mem, past_key[idx], 0,
              KV_BYTES * old_kvlen);
          d2d(bm_handle, in_tensors[k_idx + 1].device_mem, past_value[idx], 0,
              KV_BYTES * old_kvlen);
          in_tensors[k_idx].shape.dims[1] = old_kvlen;
          in_tensors[k_idx + 1].shape.dims[1] = old_kvlen;
        } else {
          // do nothing
        }
      } else {
        if (old_kvlen > 0) {
          d2d(bm_handle, in_tensors[1].device_mem, past_value[idx]);
          d2d(bm_handle, in_tensors[2].device_mem, past_key[idx]);
        } else {
          empty(bm_handle, in_tensors[1].device_mem); // recurrent state
          empty(bm_handle, in_tensors[2].device_mem); // conv state
        }
        in_tensors[0].shape.dims[1] = cur_len;
      }

      net_launch(p_bmrt, net, in_tensors, out_tensors);
      if (fa) {
        size_t offset = old_kvlen * KV_BYTES;
        bm_memcpy_d2d_byte(bm_handle, past_key[idx], offset,
                           net->stages[0].output_mems[1], 0,
                           KV_BYTES * cur_len);
        bm_memcpy_d2d_byte(bm_handle, past_value[idx], offset,
                           net->stages[0].output_mems[2], 0,
                           KV_BYTES * cur_len);
      } else {
        // reuse key as conv state
        d2d(bm_handle, past_key[idx], net->stages[0].output_mems[1]);
        // reuse value as recurrent state
        d2d(bm_handle, past_value[idx], net->stages[0].input_mems[1]);
      }
    }
    old_kvlen += cur_len;
    // D2S this chunk's output back to host result with offset
    int chunk_bytes = cur_len * HIDDEN_SIZE * sizeof(uint16_t);
    bm_memcpy_d2s_partial(bm_handle, result.data() + t * HIDDEN_SIZE, out_mem,
                          chunk_bytes);
  }
  return result;
}

ArrayUint16 Block::forward_next(ArrayInt const &position_ids,
                                ArrayUint16 &hidden_states) {
  int stage = select_decode_stage();
  int real_len = decode_stage_len.empty() ? SEQLEN : decode_stage_len[stage];

  std::vector<uint16_t> attention_mask(real_len + 1, 0);
  for (int i = history_length - 1; i < real_len; i++) {
    attention_mask[i] = mask_value;
  }
  assert(position_ids.size() == 3);
  const int *p_ids = position_ids.data();
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;

  int token_offset = (history_length - 1) * KV_BYTES;
  bm_device_mem_t out_mem;
  for (int idx = 0; idx < num_blocks; idx++) {
    int global_idx = start_idx + idx;
    bool fa = is_FA(global_idx);
    int s = fa ? stage : 0;
    if (idx == 0) {
      bm_memcpy_s2d_partial(
          bm_handle, net_blocks_cache[idx]->stages[s].input_mems[0],
          hidden_states.data(), hidden_states.size() * sizeof(uint16_t));
    } else {
      d2d(bm_handle, net_blocks_cache[idx]->stages[s].input_mems[0], out_mem);
    }
    if (fa) {
      net_launch_decode(idx, token_offset, p_ids, attention_mask, stage);
    } else {
      init_tensors(net_blocks_cache[idx], in_tensors, out_tensors);
      net_launch(p_bmrt, net_blocks_cache[idx], in_tensors, out_tensors);
    }
    out_mem = net_blocks_cache[idx]->stages[s].output_mems[0];
  }
  bm_thread_sync(bm_handle);
  int out_bytes = HIDDEN_SIZE * sizeof(uint16_t);
  ArrayUint16 results(HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, results.data(), out_mem, out_bytes);
  return results;
}

int Block::select_decode_stage() {
  if (decode_stage_len.empty()) {
    return 0;
  }
  int stage_idx = 0;
  for (auto &len : decode_stage_len) {
    if (history_length > len) {
      break;
    }
    stage_idx++;
  }
  if (stage_idx > 0) {
    stage_idx--;
  }
  return stage_idx;
}
