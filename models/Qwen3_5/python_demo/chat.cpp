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
#include <list>
#include <numeric>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <vector>

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

namespace py = pybind11;
using ArrayFloat =
    py::array_t<float, py::array::c_style | py::array::forcecast>;
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
               int stage = 0) {
  for (int i = 0; i < net->input_num; i++) {
    empty(bm_handle, net->stages[stage].input_mems[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    empty(bm_handle, net->stages[stage].output_mems[i]);
  }
}

class Qwen3_5 {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(ArrayFloat const &pixel_values, ArrayInt const &position_ids,
                   ArrayInt const &pos_idx, ArrayFloat const &pos_weight,
                   ArrayInt const &grid_thw, int vit_offset,
                   const std::string &cache_key = "");
  void forward_vit_video(ArrayFloat const &pixel_values,
                         ArrayInt const &position_ids, ArrayInt const &pos_idx,
                         ArrayFloat const &pos_weight, ArrayInt const &grid_thw,
                         ArrayInt const &vit_offsets,
                         const std::string &cache_key = "");
  int forward_first(ArrayInt const &position_ids);
  int forward_next(ArrayInt const &position_ids);
  std::vector<int> forward_next_mtp(ArrayInt const &position_ids);
  void clear_history();
  void clear_vit_cache();

  std::mt19937 sgen;
  Qwen3_5() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net,
                  const std::vector<bm_tensor_t> &in_tensors,
                  std::vector<bm_tensor_t> &out_tensors);
  void net_launch_decode(int block_idx, int kv_offset,
                         bm_device_mem_t &input_mem, const int *position_id,
                         std::vector<uint16_t> &attention_mask,
                         int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
                  int size = 0);
  void init_by_names();
  int select_decode_stage();
  int forward_first_with_kv(ArrayInt const &position_ids);
  int generate(bm_device_mem_t &logits_mem);
  void lm_head_rows(bm_device_mem_t hidden_mem, int rows, int *out_tokens);
  int lm_head_single(bm_device_mem_t hidden_mem, size_t offset_bytes,
                     bool use_mtp_head);
  void mtp_prefill(const int *pos_ids, int max_pos, int n);
  int mtp_step(bm_device_mem_t hidden_mem, size_t hidden_offset, int token,
               int pos, int slot, int visible);
  void mtp_make_drafts(int first_draft, int base_pos, int base_slot);
  int greedy_search(bm_device_mem_t &logits_mem);
  int penalty_sample(bm_device_mem_t &logits_mem);
  void init_tensors(const bm_net_info_t *net,
                    std::vector<bm_tensor_t> &in_tensors,
                    std::vector<bm_tensor_t> &out_tensors, int stage = 0);
  inline bool is_FA(int layer_idx) {
    return (layer_idx + 1) % FA_INTERVAL == 0;
  }

public:
  int token_length;
  int history_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int HIDDEN_SIZE;
  int KV_BYTES; // kv bytes for one token
  int NUM_LAYERS;
  int VIT_DIMS;
  int MAX_PATCHES;
  int MAX_PIXELS;
  int max_pos;
  bool lmhead_with_topk;
  bool has_vit = false;
  bool support_history = false;
  bool prefill_mask = false;
  uint16_t mask_value;
  std::vector<int> visited_tokens;
  const int FA_INTERVAL = 4; // full attention interval

  // MTP (llm_convert.py --mtp); greedy only, no history KV
  bool support_mtp = false;
  int MTP_K = 1;          // verify length (= draft_len + 1)
  int MTP_DRAFT_LEN = 0;  // drafts per step
  int LM_ROWS = 1;        // lm_head input rows
  int MTP_KV_BYTES = 0;   // bytes per MTP kv token
  int CONV_BYTES = 0;     // linear-attn conv state bytes
  int REC_BYTES = 0;      // linear-attn recurrent state bytes
  int mtp_length = 0;     // valid MTP kv len (slot i <-> pos i+1)
  std::vector<int> draft_tokens;
  long long mtp_steps = 0;
  long long mtp_draft_total = 0;
  long long mtp_accept_total = 0;

  // reused host scratch (avoid per-step alloc)
  std::vector<uint16_t> mask_scratch;
  std::vector<int> pos_scratch;
  std::vector<int> tokens_scratch;
  std::vector<int> cand_scratch;
  std::vector<bm_tensor_t> in_scratch;
  std::vector<bm_tensor_t> out_scratch;
  // linear-attn ping-pong: verify writes K slots, commit slot m
  std::vector<bm_device_mem_t> lin_conv_buf[2];
  std::vector<bm_device_mem_t> lin_rec_buf[2];
  int lin_parity = 0;

  // ===== ViT-output token cache =====
  struct VitCacheEntry {
    bm_device_mem_t mem;
    int num_tokens; // hw/4 image tokens this entry holds
  };
  std::unordered_map<std::string, VitCacheEntry> vit_cache;
  std::list<std::string> vit_cache_order; // FIFO eviction order
  int vit_cache_cap = 64;

  // ===== whole-video ViT cache (one entry per video) =====
  struct VideoCacheEntry {
    bm_device_mem_t mem; // t * (hw/4) * HIDDEN_SIZE * 2 bytes
    int num_frames;      // t temporal patches this entry holds
    int frame_tokens;    // hw/4 tokens per frame
  };
  std::unordered_map<std::string, VideoCacheEntry> vit_video_cache;
  std::list<std::string> vit_video_cache_order; // FIFO eviction order
  int vit_video_cache_cap = 4;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_kv;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  std::vector<int> decode_stage_len;

  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_vit;
  const bm_net_info_t *net_greedy_head, *net_sample_head;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;   // conv state or past key
  std::vector<bm_device_mem_t> past_value; // recurrent state or past value
  // MTP nets / buffers
  const bm_net_info_t *net_mtp_block = nullptr;
  const bm_net_info_t *net_mtp_cache = nullptr;
  const bm_net_info_t *net_mtp_lm = nullptr;
  std::vector<const bm_net_info_t *> net_blocks_verify;
  bm_device_mem_t mtp_past_key, mtp_past_value;
  bm_device_mem_t lm_in_buf;      // [LM_ROWS, HIDDEN]
  bm_device_mem_t mtp_hidden_buf; // last MTP hidden for chaining
  bm_device_mem_t mtp_embed_buf;  // MTP token embed / prefill staging
};

void Qwen3_5::init_tensors(const bm_net_info_t *net,
                           std::vector<bm_tensor_t> &in_tensors,
                           std::vector<bm_tensor_t> &out_tensors, int stage) {
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

void Qwen3_5::net_launch(const bm_net_info_t *net,
                         const std::vector<bm_tensor_t> &in_tensors,
                         std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
}

void Qwen3_5::net_launch_decode(int idx, int kv_offset,
                                bm_device_mem_t &input_mem, const int *pos_id,
                                std::vector<uint16_t> &attention_mask,
                                int stage_idx) {
  auto &net = net_blocks_cache[idx];
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net, in_tensors, out_tensors, stage_idx);

  // ===== prepare input tensors =====
  in_tensors[0].device_mem = input_mem;
  bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem, (void *)pos_id);
  bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                (void *)attention_mask.data());
  out_tensors[1].device_mem = bm_mem_from_device(
      past_key[idx].u.device.device_addr + kv_offset, KV_BYTES);
  out_tensors[2].device_mem = bm_mem_from_device(
      past_value[idx].u.device.device_addr + kv_offset, KV_BYTES);

  // ===== launch =====
  net_launch(net, in_tensors, out_tensors);
}

void Qwen3_5::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                  int size) {
  if (!size) {
    size = bm_mem_get_device_size(src);
  }
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Qwen3_5::clear_history() {
  mtp_length = 0;
  draft_tokens.clear();
  lin_parity = 0;
  if (!support_history) {
    return;
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
}

void Qwen3_5::init_by_names() {
  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0) {
        return true;
      }
    }
    return false;
  };
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  has_vit = is_exist("vit", net_names, num_nets);
  if (has_vit) {
    net_vit = bmrt_get_network_info(p_bmrt, "vit");
  } else {
    net_vit = nullptr;
  }
  net_greedy_head = nullptr;
  // embed, lm_head, embedding_cache, and optionally vit
  auto num_extra_nets = has_vit ? 4 : 3;
  auto num_blocks = num_nets - num_extra_nets;
  if (is_exist("greedy_head", net_names, num_nets)) {
    net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
    num_blocks--; // greedy_head is not a block
  }
  net_sample_head = nullptr;
  if (is_exist("sample_head", net_names, num_nets)) {
    net_sample_head = bmrt_get_network_info(p_bmrt, "sample_head");
    num_blocks--; // sample_head is not a block
  }
  // detect MTP nets
  support_mtp = is_exist("mtp_block", net_names, num_nets) &&
                is_exist("mtp_block_cache", net_names, num_nets);
  if (support_mtp) {
    net_mtp_block = bmrt_get_network_info(p_bmrt, "mtp_block");
    net_mtp_cache = bmrt_get_network_info(p_bmrt, "mtp_block_cache");
    num_blocks -= 2;
    if (is_exist("mtp_lm_head", net_names, num_nets)) {
      net_mtp_lm = bmrt_get_network_info(p_bmrt, "mtp_lm_head");
      num_blocks--;
    }
  }
  std::string kv_name = "block_kv_" + std::to_string(FA_INTERVAL - 1);
  if (is_exist(kv_name.c_str(), net_names, num_nets)) {
    support_history = true;
  } else {
    support_history = false;
  }
  if (support_mtp && support_history) {
    throw std::runtime_error("MTP bmodel with history KV is not supported");
  }
  // block + cache (+ verify if MTP)
  if (support_mtp) {
    NUM_LAYERS = num_blocks / 3;
  } else if (support_history) {
    NUM_LAYERS = num_blocks / (2 + 1.0 / FA_INTERVAL);
  } else {
    NUM_LAYERS = num_blocks / 2;
  }

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    // net_blocks always holds the fresh prefill net (block_); the history
    // variant (block_kv_) is collected into net_blocks_kv below.
    std::string block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    if ((!is_exist(block_name.c_str(), net_names, num_nets)) ||
        (!is_exist(cache_name.c_str(), net_names, num_nets))) {
      NUM_LAYERS = i;
      printf("Warning: Only %d blocks found, total %d blocks.\n", NUM_LAYERS,
             num_blocks);
      break;
    }
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
    if (support_history) {
      auto kv_name = "block_kv_" + std::to_string(i);
      if (is_FA(i)) {
        net_blocks_kv.emplace_back(
            bmrt_get_network_info(p_bmrt, kv_name.c_str()));
      } else {
        net_blocks_kv.emplace_back(nullptr);
      }
    }
    if (support_mtp) {
      auto verify_name = "block_verify_" + std::to_string(i);
      if (!is_exist(verify_name.c_str(), net_names, num_nets)) {
        throw std::runtime_error("MTP bmodel misses " + verify_name);
      }
      net_blocks_verify.emplace_back(
          bmrt_get_network_info(p_bmrt, verify_name.c_str()));
    }
  }
  free(net_names);
  if (net_embed_cache->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // float16
  } else if (net_embed_cache->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // -9984 by bfloat16
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }
  prefill_mask =
      net_blocks[FA_INTERVAL - 1]->input_num == (support_history ? 5 : 3);
  history_length = 0;
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;
  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[FA_INTERVAL - 1]->stages[0].input_shapes[3].dims[1];
  if (has_vit) {
    MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];
    MAX_PIXELS = MAX_PATCHES * 16 * 16;
    VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  } else {
    MAX_PATCHES = 0;
    MAX_PIXELS = 0;
    VIT_DIMS = 0;
  }
  KV_BYTES = bm_mem_get_device_size(
      net_blocks_cache[FA_INTERVAL - 1]->stages[0].output_mems[1]);
  LM_ROWS = net_lm->stages[0].input_shapes[0].dims[0];
  if (support_mtp) {
    if (!lmhead_with_topk) {
      throw std::runtime_error("MTP requires lm_head with built-in TopK "
                               "(compile without --do_sample)");
    }
    MTP_K = net_blocks_verify[0]->stages[0].input_shapes[0].dims[1];
    MTP_DRAFT_LEN = MTP_K - 1;
    if (LM_ROWS != MTP_K) {
      throw std::runtime_error("MTP: lm_head rows != verify length");
    }
    MTP_KV_BYTES =
        bm_mem_get_device_size(net_mtp_cache->stages[0].output_mems[1]);
    for (int i = 0; i < NUM_LAYERS; i++) {
      if (!is_FA(i)) {
        CONV_BYTES = bm_mem_get_device_size(
            net_blocks_cache[i]->stages[0].input_mems[1]);
        REC_BYTES = bm_mem_get_device_size(
            net_blocks_cache[i]->stages[0].input_mems[2]);
        break;
      }
    }
    if (!decode_stage_len.empty() && decode_stage_len.size() > 1) {
      throw std::runtime_error("MTP does not support chunked decode stages");
    }
    mask_scratch.reserve((size_t)SEQLEN + MTP_K + 1);
    pos_scratch.reserve(3 * (size_t)std::max(MTP_K, 1));
    tokens_scratch.reserve(std::max(MTP_K, 1));
    cand_scratch.reserve(std::max(MTP_K, 1));
  }

  printf("Num Layers:%d\n", NUM_LAYERS);
  if (has_vit) {
    printf("Max Pixels: %d*%d*%d\n", MAX_PATCHES / 4, 32, 32);
  } else {
    printf("ViT: disabled (LLM-only bmodel)\n");
  }
  PREFILL_KV_LENGTH = 0;
  if (support_history) {
    PREFILL_KV_LENGTH =
        net_blocks_kv[FA_INTERVAL - 1]->stages[0].input_shapes[3].dims[1];
    printf("History Support: True\n");
  } else {
    printf("History Support: False\n");
  }
  if (support_mtp) {
    printf("MTP Support: True (draft_len=%d, verify_len=%d, mtp_lm_head=%s)\n",
           MTP_DRAFT_LEN, MTP_K, net_mtp_lm ? "yes" : "no");
  } else {
    printf("MTP Support: False\n");
  }
}

void Qwen3_5::init(int dev_id, std::string model_path) {

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

  visited_tokens.resize(SEQLEN + MTP_K);

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    if (is_FA(i)) {
      // only init kv cache for layers with kv
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
      empty(bm_handle, past_key[i]);
      empty(bm_handle, past_value[i]);
    } else {
      // reuse key as conv state
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[1];
      // reuse value as recurrent state
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[2];
    }
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  if (!support_history) {
    auto buffer_size =
        bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
    status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
    assert(BM_SUCCESS == status);
  } else {
    // for history, we need a big buffer to store long input
    auto buffer_size = SEQLEN * HIDDEN_SIZE * sizeof(uint16_t);
    status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
    assert(BM_SUCCESS == status);
  }
  // stage single-row hidden into multi-row lm_head
  size_t row_bytes = HIDDEN_SIZE * sizeof(uint16_t);
  status = bm_malloc_device_byte(bm_handle, &lm_in_buf, LM_ROWS * row_bytes);
  assert(BM_SUCCESS == status);
  empty(bm_handle, lm_in_buf);
  if (support_mtp) {
    mtp_past_key = net_mtp_cache->stages[0].input_mems[4];
    mtp_past_value = net_mtp_cache->stages[0].input_mems[5];
    empty(bm_handle, mtp_past_key);
    empty(bm_handle, mtp_past_value);
    status = bm_malloc_device_byte(bm_handle, &mtp_hidden_buf, row_bytes);
    assert(BM_SUCCESS == status);
    status = bm_malloc_device_byte(bm_handle, &mtp_embed_buf,
                                   (size_t)MAX_INPUT_LENGTH * row_bytes);
    assert(BM_SUCCESS == status);
    if (CONV_BYTES > 0 && REC_BYTES > 0 && MTP_K > 0) {
      for (int p = 0; p < 2; p++) {
        lin_conv_buf[p].assign(NUM_LAYERS, bm_device_mem_t());
        lin_rec_buf[p].assign(NUM_LAYERS, bm_device_mem_t());
        for (int i = 0; i < NUM_LAYERS; i++) {
          if (is_FA(i)) {
            continue;
          }
          status = bm_malloc_device_byte(bm_handle, &lin_conv_buf[p][i],
                                         (size_t)MTP_K * CONV_BYTES);
          assert(BM_SUCCESS == status);
          status = bm_malloc_device_byte(bm_handle, &lin_rec_buf[p][i],
                                         (size_t)MTP_K * REC_BYTES);
          assert(BM_SUCCESS == status);
          empty(bm_handle, lin_conv_buf[p][i]);
          empty(bm_handle, lin_rec_buf[p][i]);
        }
      }
      lin_parity = 0;
    }
  }
  mtp_length = 0;
}

void Qwen3_5::clear_vit_cache() {
  for (auto &kv : vit_cache) {
    bm_free_device(bm_handle, kv.second.mem);
  }
  vit_cache.clear();
  vit_cache_order.clear();
  for (auto &kv : vit_video_cache) {
    bm_free_device(bm_handle, kv.second.mem);
  }
  vit_video_cache.clear();
  vit_video_cache_order.clear();
}

void Qwen3_5::deinit() {
  clear_vit_cache();
  bm_free_device(bm_handle, dev_buffer);
  bm_free_device(bm_handle, lm_in_buf);
  if (support_mtp) {
    bm_free_device(bm_handle, mtp_hidden_buf);
    bm_free_device(bm_handle, mtp_embed_buf);
    for (int p = 0; p < 2; p++) {
      for (auto &mem : lin_conv_buf[p]) {
        if (bm_mem_get_device_size(mem) > 0) {
          bm_free_device(bm_handle, mem);
        }
      }
      for (auto &mem : lin_rec_buf[p]) {
        if (bm_mem_get_device_size(mem) > 0) {
          bm_free_device(bm_handle, mem);
        }
      }
      lin_conv_buf[p].clear();
      lin_rec_buf[p].clear();
    }
  }
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

int Qwen3_5::greedy_search(bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_greedy_head, in_tensors, out_tensors);
  in_tensors[0].device_mem = logits_mem;
  net_launch(net_greedy_head, in_tensors, out_tensors);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_tensors[0].device_mem);
  return token;
}

void Qwen3_5::forward_embed(ArrayInt const &tokens) {
  token_length = tokens.size();
  assert(token_length <= SEQLEN);
  auto p_buffer = tokens.request();
  auto p_tokens = static_cast<int *>(p_buffer.ptr);
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(p_tokens, p_tokens + token_length, visited_tokens.data());
  if (!support_history) {
    assert(token_length <= MAX_INPUT_LENGTH);
  }
  empty(bm_handle, dev_buffer);
  for (int i = 0; i < token_length; i += MAX_INPUT_LENGTH) {
    std::vector<bm_tensor_t> in_tensors;
    std::vector<bm_tensor_t> out_tensors;
    init_tensors(net_embed, in_tensors, out_tensors);
    int real_len = std::min(MAX_INPUT_LENGTH, token_length - i);
    if (real_len != MAX_INPUT_LENGTH) {
      empty(bm_handle, in_tensors[0].device_mem);
    }
    bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                          (void *)(visited_tokens.data() + i),
                          real_len * sizeof(int));
    net_launch(net_embed, in_tensors, out_tensors);
    int offset = i * HIDDEN_SIZE * sizeof(uint16_t);
    d2d(dev_buffer, out_tensors[0].device_mem, offset,
        real_len * HIDDEN_SIZE * sizeof(uint16_t));
  }
}

void Qwen3_5::forward_vit(ArrayFloat const &pixel_values,
                          ArrayInt const &position_ids, ArrayInt const &pos_idx,
                          ArrayFloat const &pos_weight,
                          ArrayInt const &grid_thw, int vit_offset,
                          const std::string &cache_key) {
  if (!has_vit) {
    throw std::runtime_error(
        "forward_vit: bmodel has no vit network (LLM-only mode)");
  }
  auto p_grid_thw = grid_thw.request();
  auto p_thw = static_cast<int *>(p_grid_thw.ptr);
  int t = p_thw[0];
  int h = p_thw[1];
  int w = p_thw[2];
  int hw = t * h * w;
  int num_pixels = hw * VIT_DIMS;
  assert(pixel_values.size() == num_pixels);
  assert(position_ids.size() == (hw * 2));
  assert(pos_idx.size() == (hw * 4));
  assert(pos_weight.size() == (hw * 4));
  auto p_pixel_values = pixel_values.request();
  auto p_position_ids = position_ids.request();
  auto p_pos_idx = pos_idx.request();
  auto p_pos_weight = pos_weight.request();

  // concatenate text embedding and image embedding
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = hw / 4 * HIDDEN_SIZE * sizeof(uint16_t);

  // ===== cache HIT: skip ViT net_launch, just place cached embeddings =====
  if (!cache_key.empty()) {
    auto it = vit_cache.find(cache_key);
    if (it != vit_cache.end()) {
      assert(it->second.num_tokens == hw / 4);
      bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                         it->second.mem, 0,
                         it->second.num_tokens * HIDDEN_SIZE *
                             sizeof(uint16_t));
      return;
    }
  }

  // ===== cache MISS: run the ViT net as usual =====
  empty_net(bm_handle, net_vit);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_vit, in_tensors, out_tensors);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)p_pixel_values.ptr, num_pixels * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                        (void *)p_position_ids.ptr,
                        position_ids.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                        (void *)p_pos_idx.ptr, pos_idx.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[3].device_mem,
                        (void *)p_pos_weight.ptr,
                        pos_weight.size() * sizeof(float));
  in_tensors[0].shape.dims[0] = hw;
  in_tensors[1].shape.dims[0] = hw;
  in_tensors[2].shape.dims[0] = hw;
  in_tensors[3].shape.dims[0] = hw;
  net_launch(net_vit, in_tensors, out_tensors);

  // place the freshly computed image embeddings into dev_buffer (as before)
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                     out_tensors[0].device_mem, 0, vit_size);

  // ===== store the result into the ViT cache for future reuse =====
  if (!cache_key.empty() && vit_cache_cap > 0) {
    auto it = vit_cache.find(cache_key);
    if (it != vit_cache.end()) {
      bm_free_device(bm_handle, it->second.mem);
      vit_cache_order.remove(cache_key);
      vit_cache.erase(it);
    } else if ((int)vit_cache.size() >= vit_cache_cap) {
      // FIFO eviction of the oldest entry
      std::string oldest = vit_cache_order.front();
      bm_free_device(bm_handle, vit_cache[oldest].mem);
      vit_cache.erase(oldest);
      vit_cache_order.pop_front();
    }
    VitCacheEntry entry;
    auto ret = bm_malloc_device_byte(bm_handle, &entry.mem, vit_size);
    assert(BM_SUCCESS == ret);
    bm_memcpy_d2d_byte(bm_handle, entry.mem, 0, out_tensors[0].device_mem, 0,
                       vit_size);
    entry.num_tokens = hw / 4;
    vit_cache[cache_key] = entry;
    vit_cache_order.push_back(cache_key);
  }
}

void Qwen3_5::forward_vit_video(ArrayFloat const &pixel_values,
                                ArrayInt const &position_ids,
                                ArrayInt const &pos_idx,
                                ArrayFloat const &pos_weight,
                                ArrayInt const &grid_thw,
                                ArrayInt const &vit_offsets,
                                const std::string &cache_key) {
  if (!has_vit) {
    throw std::runtime_error(
        "forward_vit_video: bmodel has no vit network (LLM-only mode)");
  }
  auto p_grid_thw = grid_thw.request();
  auto p_thw = static_cast<int *>(p_grid_thw.ptr);
  assert(p_thw[0] == 1); // each video segment is a single-frame {1,h,w} grid
  int h = p_thw[1];
  int w = p_thw[2];
  int hw = h * w;
  int num_frames = (int)vit_offsets.size();
  int num_pixels = hw * VIT_DIMS;
  assert((int)position_ids.size() == (hw * 2));
  assert((int)pos_idx.size() == (hw * 4));
  assert((int)pos_weight.size() == (hw * 4));
  assert((int)pixel_values.size() == num_frames * num_pixels);
  auto p_pixel_values = pixel_values.request();
  auto p_position_ids = position_ids.request();
  auto p_pos_idx = pos_idx.request();
  auto p_pos_weight = pos_weight.request();
  auto p_offsets = vit_offsets.request();
  auto p_vit_offsets = static_cast<int *>(p_offsets.ptr);
  auto pixels = static_cast<const float *>(p_pixel_values.ptr);

  int frame_size = hw / 4 * HIDDEN_SIZE * sizeof(uint16_t);
  size_t total_size = (size_t)frame_size * num_frames;

  // ===== cache HIT: skip every per-patch net_launch, replay cached bytes =====
  if (!cache_key.empty()) {
    auto it = vit_video_cache.find(cache_key);
    if (it != vit_video_cache.end()) {
      // The key encodes {h, w, t}, so the entry must hold exactly this many
      // frames/tokens; assert to catch any stale/corrupted entry loudly.
      assert(it->second.num_frames == num_frames);
      assert(it->second.frame_tokens == hw / 4);
      for (int i = 0; i < num_frames; ++i) {
        int dst_offset = p_vit_offsets[i] * HIDDEN_SIZE * sizeof(uint16_t);
        bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset, it->second.mem,
                           (size_t)i * frame_size, frame_size);
      }
      return;
    }
  }

  // ===== cache MISS: run the ViT once per patch (as before). The entry is
  // allocated up front so each patch's output can be snapshotted right after
  // its launch — the net's output device mem is reused by the next patch.
  // =====
  bool store = !cache_key.empty() && vit_video_cache_cap > 0;
  bm_device_mem_t entry_mem;
  if (store) {
    // defensive: if a stale entry under this key exists, free it first
    auto it = vit_video_cache.find(cache_key);
    if (it != vit_video_cache.end()) {
      bm_free_device(bm_handle, it->second.mem);
      vit_video_cache_order.remove(cache_key);
      vit_video_cache.erase(it);
    } else if ((int)vit_video_cache.size() >= vit_video_cache_cap) {
      // FIFO eviction of the oldest entry
      std::string oldest = vit_video_cache_order.front();
      bm_free_device(bm_handle, vit_video_cache[oldest].mem);
      vit_video_cache.erase(oldest);
      vit_video_cache_order.pop_front();
    }
    auto ret = bm_malloc_device_byte(bm_handle, &entry_mem, total_size);
    if (ret != BM_SUCCESS) {
      store = false;
    }
  }
  for (int i = 0; i < num_frames; ++i) {
    empty_net(bm_handle, net_vit);
    std::vector<bm_tensor_t> in_tensors;
    std::vector<bm_tensor_t> out_tensors;
    init_tensors(net_vit, in_tensors, out_tensors);
    bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                          (void *)(pixels + (size_t)i * num_pixels),
                          num_pixels * sizeof(float));
    bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                          (void *)p_position_ids.ptr,
                          position_ids.size() * sizeof(int));
    bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                          (void *)p_pos_idx.ptr, pos_idx.size() * sizeof(int));
    bm_memcpy_s2d_partial(bm_handle, in_tensors[3].device_mem,
                          (void *)p_pos_weight.ptr,
                          pos_weight.size() * sizeof(float));
    in_tensors[0].shape.dims[0] = hw;
    in_tensors[1].shape.dims[0] = hw;
    in_tensors[2].shape.dims[0] = hw;
    in_tensors[3].shape.dims[0] = hw;
    net_launch(net_vit, in_tensors, out_tensors);

    int dst_offset = p_vit_offsets[i] * HIDDEN_SIZE * sizeof(uint16_t);
    // place the freshly computed patch embeddings into dev_buffer (as before)
    bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                       out_tensors[0].device_mem, 0, frame_size);
    if (store) {
      bm_memcpy_d2d_byte(bm_handle, entry_mem, (size_t)i * frame_size,
                         out_tensors[0].device_mem, 0, frame_size);
    }
  }
  if (store) {
    VideoCacheEntry entry;
    entry.mem = entry_mem;
    entry.num_frames = num_frames;
    entry.frame_tokens = hw / 4;
    vit_video_cache[cache_key] = entry;
    vit_video_cache_order.push_back(cache_key);
  }
}

int Qwen3_5::generate(bm_device_mem_t &logits_mem) {
  int token = 0;
  if (lmhead_with_topk) {
    bm_memcpy_d2s_partial(bm_handle, (void *)&token, logits_mem, sizeof(int));
  } else {
    token = greedy_search(logits_mem);
  }
  return token;
}

//===------------------------------------------------------------===//
// MTP helpers
//===------------------------------------------------------------===//
// Score `rows` hidden rows with trunk lm_head (built-in TopK).
void Qwen3_5::lm_head_rows(bm_device_mem_t hidden_mem, int rows,
                           int *out_tokens) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = hidden_mem;
  net_launch(net_lm, in_tensors, out_tensors);
  std::vector<int> buf(LM_ROWS, 0);
  bm_memcpy_d2s_partial(bm_handle, (void *)buf.data(),
                        out_tensors[0].device_mem, LM_ROWS * sizeof(int));
  std::copy(buf.begin(), buf.begin() + rows, out_tokens);
}

// Greedy token for one hidden row; optional truncated mtp_lm_head.
int Qwen3_5::lm_head_single(bm_device_mem_t hidden_mem, size_t offset_bytes,
                            bool use_mtp_head) {
  size_t row_bytes = HIDDEN_SIZE * sizeof(uint16_t);
  if (use_mtp_head && net_mtp_lm != nullptr) {
    std::vector<bm_tensor_t> in_tensors;
    std::vector<bm_tensor_t> out_tensors;
    init_tensors(net_mtp_lm, in_tensors, out_tensors);
    in_tensors[0].device_mem = bm_mem_from_device(
        hidden_mem.u.device.device_addr + offset_bytes, row_bytes);
    net_launch(net_mtp_lm, in_tensors, out_tensors);
    int token = 0;
    bm_memcpy_d2s_partial(bm_handle, (void *)&token,
                          out_tensors[0].device_mem, sizeof(int));
    return token;
  }
  bm_memcpy_d2d_byte(bm_handle, lm_in_buf, 0, hidden_mem, offset_bytes,
                     row_bytes);
  int token = 0;
  lm_head_rows(lm_in_buf, 1, &token);
  return token;
}

// One MTP step: hidden@pos-1 + token@pos -> draft@pos+1; write kv to `slot`.
int Qwen3_5::mtp_step(bm_device_mem_t hidden_mem, size_t hidden_offset,
                      int token, int pos, int slot, int visible) {
  size_t row_bytes = HIDDEN_SIZE * sizeof(uint16_t);
  in_scratch.clear();
  out_scratch.clear();
  init_tensors(net_embed_cache, in_scratch, out_scratch);
  bm_memcpy_s2d(bm_handle, in_scratch[0].device_mem, (void *)&token);
  net_launch(net_embed_cache, in_scratch, out_scratch);
  d2d(mtp_embed_buf, out_scratch[0].device_mem, 0, row_bytes);
  in_scratch.clear();
  out_scratch.clear();
  init_tensors(net_mtp_cache, in_scratch, out_scratch);
  in_scratch[0].device_mem = bm_mem_from_device(
      hidden_mem.u.device.device_addr + hidden_offset, row_bytes);
  in_scratch[1].device_mem = mtp_embed_buf;
  int pos3[3] = {pos, pos, pos};
  bm_memcpy_s2d(bm_handle, in_scratch[2].device_mem, (void *)pos3);
  mask_scratch.assign(SEQLEN + 1, mask_value);
  for (int i = 0; i < visible && i < SEQLEN; i++) {
    mask_scratch[i] = 0;
  }
  mask_scratch[SEQLEN] = 0;
  bm_memcpy_s2d(bm_handle, in_scratch[3].device_mem,
                (void *)mask_scratch.data());
  in_scratch[4].device_mem = mtp_past_key;
  in_scratch[5].device_mem = mtp_past_value;
  out_scratch[1].device_mem = bm_mem_from_device(
      mtp_past_key.u.device.device_addr + (size_t)slot * MTP_KV_BYTES,
      MTP_KV_BYTES);
  out_scratch[2].device_mem = bm_mem_from_device(
      mtp_past_value.u.device.device_addr + (size_t)slot * MTP_KV_BYTES,
      MTP_KV_BYTES);
  net_launch(net_mtp_cache, in_scratch, out_scratch);
  return lm_head_single(out_scratch[0].device_mem, 0, true);
}

// Chain-generate MTP_DRAFT_LEN drafts from first_draft + mtp_hidden_buf.
void Qwen3_5::mtp_make_drafts(int first_draft, int base_pos, int base_slot) {
  size_t row_bytes = HIDDEN_SIZE * sizeof(uint16_t);
  draft_tokens.assign(1, first_draft);
  for (int i = 1; i < MTP_DRAFT_LEN; i++) {
    int slot = base_slot + i - 1;
    if (slot >= SEQLEN) {
      draft_tokens.push_back(draft_tokens.back());
      continue;
    }
    int draft = mtp_step(mtp_hidden_buf, 0, draft_tokens.back(), base_pos + i,
                         slot, slot);
    bm_memcpy_d2d_byte(bm_handle, mtp_hidden_buf, 0,
                       net_mtp_cache->stages[0].output_mems[0], 0, row_bytes);
    draft_tokens.push_back(draft);
  }
}

// After trunk prefill: fill MTP kv with trunk hidden[0,n) + embed[1,n].
void Qwen3_5::mtp_prefill(const int *pos_ids, int max_pos, int n) {
  size_t row_bytes = HIDDEN_SIZE * sizeof(uint16_t);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed, in_tensors, out_tensors);
  empty(bm_handle, in_tensors[0].device_mem);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)(visited_tokens.data() + 1), n * sizeof(int));
  net_launch(net_embed, in_tensors, out_tensors);
  auto embed_mem = out_tensors[0].device_mem;
  // token n sits at max_pos + 1
  std::vector<int> mpos(3 * n, 0);
  for (int r = 0; r < 3; r++) {
    for (int i = 0; i + 1 < n; i++) {
      mpos[r * n + i] = pos_ids[r * n + i + 1];
    }
    mpos[r * n + n - 1] = max_pos + 1;
  }
  // stage embed out of shared neuron pool before launching mtp_block
  d2d(mtp_embed_buf, embed_mem, 0, n * row_bytes);
  init_tensors(net_mtp_block, in_tensors, out_tensors);
  d2d(in_tensors[0].device_mem, dev_buffer, 0, n * row_bytes);
  d2d(in_tensors[1].device_mem, mtp_embed_buf, 0, n * row_bytes);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                        (void *)mpos.data(), 3 * n * sizeof(int));
  in_tensors[0].shape.dims[1] = n;
  in_tensors[1].shape.dims[1] = n;
  in_tensors[2].shape.dims[1] = n;
  if (net_mtp_block->input_num > 3) {
    std::vector<uint16_t> attention_mask(n * n, mask_value);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * n + j] = 0;
      }
    }
    bm_memcpy_s2d_partial(bm_handle, in_tensors[3].device_mem,
                          (void *)attention_mask.data(),
                          n * n * sizeof(uint16_t));
    in_tensors[3].shape.dims[2] = n;
    in_tensors[3].shape.dims[3] = n;
  }
  net_launch(net_mtp_block, in_tensors, out_tensors);
  bm_memcpy_d2d_byte(bm_handle, mtp_past_key, 0,
                     net_mtp_block->stages[0].output_mems[1], 0,
                     (size_t)MTP_KV_BYTES * n);
  bm_memcpy_d2d_byte(bm_handle, mtp_past_value, 0,
                     net_mtp_block->stages[0].output_mems[2], 0,
                     (size_t)MTP_KV_BYTES * n);
  mtp_length = n;
  // copy last MTP hidden before lm_head may overwrite shared mem
  auto mtp_out = net_mtp_block->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, mtp_hidden_buf, 0, mtp_out,
                     (size_t)(n - 1) * row_bytes, row_bytes);
  int draft = lm_head_single(mtp_hidden_buf, 0, true);
  mtp_make_drafts(draft, max_pos + 1, n);
}

int Qwen3_5::forward_first(ArrayInt const &position_ids) {
  if (support_history) {
    return forward_first_with_kv(position_ids);
  }
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  std::vector<int> position_ids_pad;
  std::vector<uint16_t> attention_mask;
  if (prefill_mask) {
    attention_mask.resize(token_length * token_length, mask_value);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * token_length + j] = 0;
      }
    }
  }
  position_ids_pad.assign(3 * token_length, 0);
  assert((int)position_ids.size() == token_length * 3);
  std::copy(p_ids, p_ids + token_length * 3, position_ids_pad.begin());

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors);
    out_tensors[0].device_mem = out_mem;
    d2d(in_tensors[0].device_mem, out_mem, 0,
        token_length * HIDDEN_SIZE * sizeof(uint16_t));
    if (is_FA(idx)) {
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
      in_tensors[0].shape.dims[1] = token_length;
      empty(bm_handle, in_tensors[1].device_mem); // recurrent state
    }

    net_launch(net_blocks[idx], in_tensors, out_tensors);
    if (is_FA(idx)) {
      bm_memcpy_d2d_byte(bm_handle, past_key[idx], 0,
                         net_blocks[idx]->stages[0].output_mems[1], 0,
                         KV_BYTES * token_length);
      bm_memcpy_d2d_byte(bm_handle, past_value[idx], 0,
                         net_blocks[idx]->stages[0].output_mems[2], 0,
                         KV_BYTES * token_length);
    } else {
      // reuse key as conv state
      d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
      // reuse value as recurrent state
      d2d(past_value[idx], net_blocks[idx]->stages[0].input_mems[1]);
    }
  }

  // forward lmhead
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  int token = 0;
  if (LM_ROWS > 1 || support_mtp) {
    // keep trunk hidden in dev_buffer for mtp_prefill
    token = lm_head_single(out_mem, (size_t)(token_length - 1) * bytes, false);
  } else {
    init_tensors(net_lm, in_tensors, out_tensors);
    in_tensors[0].device_mem = bm_mem_from_device(
        out_mem.u.device.device_addr + (token_length - 1) * bytes, bytes);
    out_tensors[0].device_mem = dev_buffer;
    net_launch(net_lm, in_tensors, out_tensors);
    token = generate(dev_buffer);
  }
  visited_tokens[token_length] = token;
  if (support_mtp) {
    int max_pos = 0;
    for (int i = 0; i < 3 * token_length; i++) {
      max_pos = std::max(max_pos, position_ids_pad[i]);
    }
    mtp_prefill(position_ids_pad.data(), max_pos, token_length);
  }
  token_length++;
  history_length = token_length;
  return token;
}

int Qwen3_5::forward_first_with_kv(ArrayInt const &position_ids) {
  assert(history_length + token_length < SEQLEN);
  assert(prefill_mask == false);
  assert((int)position_ids.size() == 3 * token_length);
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  std::vector<int> pos_ids(3 * MAX_INPUT_LENGTH, 0);

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  int k_idx = prefill_mask ? 3 : 2;
  int old_kvlen = (history_length > 0) ? (history_length - 1) : 0;
  int last_cur_len = 0; // cur_len of the last chunk, for lmhead offset
  for (int t = 0; t < token_length; t += MAX_INPUT_LENGTH) {
    auto old_length = history_length;
    int cur_len = std::min(MAX_INPUT_LENGTH, token_length - t);
    last_cur_len = cur_len;
    history_length += cur_len;
    // copy position ids with offset
    for (int i = 0; i < 3; i++) {
      std::copy(p_ids + i * token_length + t,
                p_ids + i * token_length + t + cur_len,
                pos_ids.data() + i * cur_len);
    }

    assert(old_length <= PREFILL_KV_LENGTH);
    for (int idx = 0; idx < NUM_LAYERS; idx++) {
      auto fa = is_FA(idx);
      auto &net = (old_kvlen > 0 && fa) ? net_blocks_kv[idx] : net_blocks[idx];
      init_tensors(net, in_tensors, out_tensors);
      out_tensors[0].device_mem = out_mem;

      if (idx == 0) {
        // for the first block, we need to copy input from dev_buffer to
        // in_tensors[0] with offset
        bm_memcpy_d2d_byte(bm_handle, in_tensors[0].device_mem, 0, dev_buffer,
                           t * HIDDEN_SIZE * sizeof(uint16_t),
                           cur_len * HIDDEN_SIZE * sizeof(uint16_t));
      } else {
        d2d(in_tensors[0].device_mem, out_mem, 0,
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
          d2d(in_tensors[k_idx].device_mem, past_key[idx], 0,
              KV_BYTES * old_kvlen);
          d2d(in_tensors[k_idx + 1].device_mem, past_value[idx], 0,
              KV_BYTES * old_kvlen);
          in_tensors[k_idx].shape.dims[1] = old_kvlen;
          in_tensors[k_idx + 1].shape.dims[1] = old_kvlen;
        } else {
          // do nothing
        }
      } else {
        if (old_kvlen > 0) {
          d2d(in_tensors[1].device_mem, past_value[idx]);
          d2d(in_tensors[2].device_mem, past_key[idx]);
        } else {
          empty(bm_handle, in_tensors[1].device_mem); // recurrent state
          empty(bm_handle, in_tensors[2].device_mem); // conv state
        }
        in_tensors[0].shape.dims[1] = cur_len;
      }

      net_launch(net, in_tensors, out_tensors);
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
        d2d(past_key[idx], net->stages[0].output_mems[1]);
        // reuse value as recurrent state
        d2d(past_value[idx], net->stages[0].input_mems[1]);
      }
    }
    old_kvlen += cur_len;
  }
  // forward lmhead
  // the last chunk writes its block output to dev_buffer[0..last_cur_len),
  // so the last token's hidden state is at (last_cur_len - 1) * bytes,
  // not (token_length - 1) * bytes (which is stale/garbage when chunked).
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = bm_mem_from_device(
      out_mem.u.device.device_addr + (last_cur_len - 1) * bytes, bytes);
  out_tensors[0].device_mem = dev_buffer;
  net_launch(net_lm, in_tensors, out_tensors);
  int token = generate(dev_buffer);
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

int Qwen3_5::select_decode_stage() {
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

int Qwen3_5::forward_next(ArrayInt const &position_ids) {
  int stage = select_decode_stage();
  int real_len = decode_stage_len.empty() ? SEQLEN : decode_stage_len[stage];

  mask_scratch.assign(real_len + 1, 0);
  for (int i = history_length - 1; i < real_len; i++) {
    mask_scratch[i] = mask_value;
  }
  assert(position_ids.size() == 3);
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  // embedding
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed_cache, in_tensors, out_tensors);
  int token = visited_tokens[token_length - 1];
  bm_memcpy_s2d(bm_handle, in_tensors[0].device_mem, (void *)&token);
  net_launch(net_embed_cache, in_tensors, out_tensors);
  auto out_mem = out_tensors[0].device_mem;

  // blocks
  int fa_bytes = bm_mem_get_device_size(
      net_blocks_cache[FA_INTERVAL - 1]->stages[0].output_mems[1]);
  int token_offset = (history_length - 1) * fa_bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    if (is_FA(idx)) {
      net_launch_decode(idx, token_offset, out_mem, p_ids, mask_scratch,
                        stage);
      out_mem = net_blocks_cache[idx]->stages[stage].output_mems[0];
    } else {
      init_tensors(net_blocks_cache[idx], in_tensors, out_tensors);
      in_tensors[0].device_mem = out_mem;
      net_launch(net_blocks_cache[idx], in_tensors, out_tensors);
      out_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    }
  }

  // forward lmhead
  if (LM_ROWS > 1) {
    token = lm_head_single(out_mem, 0, false);
  } else {
    init_tensors(net_lm, in_tensors, out_tensors);
    in_tensors[0].device_mem = out_mem;
    out_tensors[0].device_mem = dev_buffer;
    net_launch(net_lm, in_tensors, out_tensors);
    token = generate(dev_buffer);
  }
  visited_tokens[token_length] = token;
  if (support_mtp) {
    // sync MTP kv after single-token decode
    int n = history_length - 1;
    if (mtp_length == n && n < SEQLEN) {
      int draft = mtp_step(out_mem, 0, token, p_ids[0] + 1, n, n);
      bm_memcpy_d2d_byte(bm_handle, mtp_hidden_buf, 0,
                         net_mtp_cache->stages[0].output_mems[0], 0,
                         HIDDEN_SIZE * sizeof(uint16_t));
      mtp_length = n + 1;
      mtp_make_drafts(draft, p_ids[0] + 1, n + 1);
    } else {
      mtp_length = -1;
      draft_tokens.clear();
    }
  }
  token_length++;
  history_length++;
  return token;
}

// Verify [t_n, drafts...] once; accept longest matching prefix (1..MTP_K).
std::vector<int> Qwen3_5::forward_next_mtp(ArrayInt const &position_ids) {
  assert(position_ids.size() == 3);
  int n = token_length - 1;
  if (!support_mtp || mtp_length != n ||
      (int)draft_tokens.size() != MTP_DRAFT_LEN || n + MTP_K > SEQLEN) {
    return {forward_next(position_ids)};
  }
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  size_t row_bytes = HIDDEN_SIZE * sizeof(uint16_t);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;

  // 1. embed [t_n, drafts...]
  tokens_scratch.resize(MTP_K);
  tokens_scratch[0] = visited_tokens[n];
  std::copy(draft_tokens.begin(), draft_tokens.end(), tokens_scratch.begin() + 1);
  init_tensors(net_embed, in_tensors, out_tensors);
  empty(bm_handle, in_tensors[0].device_mem);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)tokens_scratch.data(), MTP_K * sizeof(int));
  net_launch(net_embed, in_tensors, out_tensors);
  d2d(dev_buffer, out_tensors[0].device_mem, 0, MTP_K * row_bytes);
  auto out_mem = dev_buffer;

  // 2. positions + causal mask over history and the K-token batch
  pos_scratch.resize(3 * MTP_K);
  for (int r = 0; r < 3; r++) {
    for (int j = 0; j < MTP_K; j++) {
      pos_scratch[r * MTP_K + j] = p_ids[r] + j;
    }
  }
  int mask_w = SEQLEN + MTP_K;
  mask_scratch.resize((size_t)MTP_K * mask_w);
  std::fill(mask_scratch.begin(), mask_scratch.end(), mask_value);
  for (int j = 0; j < MTP_K; j++) {
    for (int i = 0; i < n; i++) {
      mask_scratch[(size_t)j * mask_w + i] = 0;
    }
    for (int c = 0; c <= j; c++) {
      mask_scratch[(size_t)j * mask_w + SEQLEN + c] = 0;
    }
  }

  // 3. trunk verify
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &net = net_blocks_verify[idx];
    init_tensors(net, in_tensors, out_tensors);
    in_tensors[0].device_mem = out_mem;
    if (is_FA(idx)) {
      bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                    (void *)pos_scratch.data());
      bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                    (void *)mask_scratch.data());
      in_tensors[3].device_mem = past_key[idx];
      in_tensors[4].device_mem = past_value[idx];
      // write new kv at [n, n+K); rejected tail is overwritten next step
      out_tensors[1].device_mem = bm_mem_from_device(
          past_key[idx].u.device.device_addr + (size_t)n * KV_BYTES,
          (size_t)KV_BYTES * MTP_K);
      out_tensors[2].device_mem = bm_mem_from_device(
          past_value[idx].u.device.device_addr + (size_t)n * KV_BYTES,
          (size_t)KV_BYTES * MTP_K);
    } else {
      // write K candidate states to idle ping-pong buffer
      if (lin_conv_buf[0].empty() ||
          bm_mem_get_device_size(lin_conv_buf[lin_parity][idx]) == 0) {
        throw std::runtime_error(
            "MTP linear-attn ping-pong buffers are not allocated");
      }
      in_tensors[1].device_mem = past_key[idx];
      in_tensors[2].device_mem = past_value[idx];
      out_tensors[1].device_mem = lin_conv_buf[lin_parity][idx];
      out_tensors[2].device_mem = lin_rec_buf[lin_parity][idx];
    }
    net_launch(net, in_tensors, out_tensors);
    out_mem = net->stages[0].output_mems[0];
  }

  // 4. accept longest draft prefix that matches trunk greedy tokens
  cand_scratch.resize(MTP_K);
  lm_head_rows(out_mem, MTP_K, cand_scratch.data());
  int m = 0;
  while (m < MTP_DRAFT_LEN && cand_scratch[m] == draft_tokens[m]) {
    m++;
  }

  // 5. commit linear-attn state at slot m; flip ping-pong
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    if (is_FA(idx)) {
      continue;
    }
    past_key[idx] = bm_mem_from_device(
        lin_conv_buf[lin_parity][idx].u.device.device_addr +
            (size_t)m * CONV_BYTES,
        CONV_BYTES);
    past_value[idx] = bm_mem_from_device(
        lin_rec_buf[lin_parity][idx].u.device.device_addr +
            (size_t)m * REC_BYTES,
        REC_BYTES);
  }
  if (!lin_conv_buf[0].empty()) {
    lin_parity ^= 1;
  }

  // 6. bookkeeping
  std::vector<int> accepted(cand_scratch.begin(), cand_scratch.begin() + m + 1);
  for (int j = 0; j <= m; j++) {
    visited_tokens[token_length + j] = cand_scratch[j];
  }
  token_length += m + 1;
  history_length = token_length;
  int n2 = n + m + 1;
  mtp_steps++;
  mtp_draft_total += MTP_DRAFT_LEN;
  mtp_accept_total += m;

  // 7. refresh MTP with true trunk hiddens, then draft next step
  int draft = 0;
  for (int j = 0; j <= m; j++) {
    draft = mtp_step(out_mem, (size_t)j * row_bytes, cand_scratch[j], p_ids[0] + j + 1,
                     n + j, n + j);
  }
  bm_memcpy_d2d_byte(bm_handle, mtp_hidden_buf, 0,
                     net_mtp_cache->stages[0].output_mems[0], 0, row_bytes);
  mtp_length = n2;
  mtp_make_drafts(draft, p_ids[0] + m + 1, n2);
  return accepted;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen3_5>(m, "Qwen3_5")
      .def(pybind11::init<>())
      .def("init", &Qwen3_5::init)
      .def("forward_embed", &Qwen3_5::forward_embed)
      .def("forward_vit", &Qwen3_5::forward_vit, py::arg("pixel_values"),
           py::arg("position_ids"), py::arg("pos_idx"), py::arg("pos_weight"),
           py::arg("grid_thw"), py::arg("vit_offset"),
           py::arg("cache_key") = std::string(""))
      .def("forward_vit_video", &Qwen3_5::forward_vit_video,
           py::arg("pixel_values"), py::arg("position_ids"), py::arg("pos_idx"),
           py::arg("pos_weight"), py::arg("grid_thw"), py::arg("vit_offsets"),
           py::arg("cache_key") = std::string(""))
      .def("forward_first", &Qwen3_5::forward_first)
      .def("forward_next", &Qwen3_5::forward_next)
      .def("forward_next_mtp", &Qwen3_5::forward_next_mtp)
      .def("clear_history", &Qwen3_5::clear_history)
      .def("clear_vit_cache", &Qwen3_5::clear_vit_cache)
      .def("deinit", &Qwen3_5::deinit)
      .def_readwrite("vit_cache_cap", &Qwen3_5::vit_cache_cap)
      .def_readwrite("vit_video_cache_cap", &Qwen3_5::vit_video_cache_cap)
      .def_readonly("SEQLEN", &Qwen3_5::SEQLEN) // read SEQLEN in pipeline.py
      .def_readonly("MAX_PIXELS", &Qwen3_5::MAX_PIXELS)
      .def_readonly("MAX_PATCHES", &Qwen3_5::MAX_PATCHES)
      .def_readonly("MAX_INPUT_LENGTH", &Qwen3_5::MAX_INPUT_LENGTH)
      .def_readonly("PREFILL_KV_LENGTH", &Qwen3_5::PREFILL_KV_LENGTH)
      .def_readonly("support_history", &Qwen3_5::support_history)
      .def_readonly("history_length", &Qwen3_5::history_length)
      .def_readonly("has_vit", &Qwen3_5::has_vit)
      .def_readonly("support_mtp", &Qwen3_5::support_mtp)
      .def_readonly("MTP_K", &Qwen3_5::MTP_K)
      .def_readonly("MTP_DRAFT_LEN", &Qwen3_5::MTP_DRAFT_LEN)
      .def_readonly("mtp_steps", &Qwen3_5::mtp_steps)
      .def_readonly("mtp_draft_total", &Qwen3_5::mtp_draft_total)
      .def_readonly("mtp_accept_total", &Qwen3_5::mtp_accept_total);
}
