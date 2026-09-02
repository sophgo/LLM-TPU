//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// Mage-VL pybind11 runtime (phase 1 offline VLM + phase 4 streaming gate).
//
// Text backbone  : plain Qwen3-4B (1D RoPE, QK-norm, GQA). Inherited from the
//                 bmodel's standard embedding/block/block_cache/lm_head nets,
//                 so forward_embed / forward_first / forward_next mirror the
//                 Qwen3 text demo (1D position_ids, static prefill mask).
// Vision tower   : custom Mage-ViT compiled as a single "vit" net with a
//                 5-input contract:
//                   in[0] input_states   [N, 768]   F32   (flattened patches)
//                   in[1] position_ids   [N]        INT32 (pos_t)
//                   in[2] position_ids   [N]        INT32 (pos_h)
//                   in[3] position_ids   [N]        INT32 (pos_w)
//                   in[4] attention_mask  [1,1,N,N] F32   (bidirectional)
//                   out[0] merger.mlp.2  [N/4, 2560] BF16 (== HIDDEN_SIZE)
//                 The 3D RoPE lives INSIDE the vit net (baked cos/sin tables +
//                 permutation matrix), so the LLM stays on plain 1D RoPE.
// Gate + ClsNet  : StreamMind streaming decision controller. Two small bmodel
//                 nets compiled into the same combined bmodel:
//                   gate    : [1, T, 2560] BF16 -> [1, T, 2560] BF16
//                             (PreNet + VideoMamba + PostNet)
//                   cls_net : [1, T, 2560] BF16 -> [1, T, 2] BF16
//                             (4-layer Qwen3 binary classifier: silent/speak)
//                 T is GATE_FRAMES (static, extracted from gate input shape).
//
// Flow: forward_embed(tokens) -> dev_buffer; forward_vit(...) injects the
//       merged image embeddings into dev_buffer at the image_pad span;
//       forward_first(position_ids) prefills over dev_buffer; forward_next()
//       decodes (1D position_id = history_length-1, computed internally).
//
// Streaming: for each video segment of T frames, call forward_vit per frame,
//       then forward_gate(averaged_tokens) -> logits [T, 2]. Python decides
//       whether to speak from the logits (e.g. argmax or threshold on col 1).
//===----------------------------------------------------------------------===//

#include "bmruntime_interface.h"
#include "memory.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
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

// Convert bfloat16 (stored as uint16_t) to float32.
static inline float bf16_to_f32(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f;
  memcpy(&f, &bits, sizeof(float));
  return f;
}

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

class Mage_VL {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(ArrayFloat const &pixel_values, ArrayInt const &pos_t,
                   ArrayInt const &pos_h, ArrayInt const &pos_w,
                   int vit_offset);
  int forward_first(ArrayInt const &position_ids);
  int forward_next();
  void clear_history();
  std::vector<float> forward_gate(ArrayFloat const &averaged_tokens);
  py::array_t<float> read_vit_embeddings(int offset, int num_tokens);

  std::mt19937 sgen;
  Mage_VL() : sgen(std::random_device()()), p_bmrt(nullptr) {}

private:
  void net_launch(const bm_net_info_t *net,
                  const std::vector<bm_tensor_t> &in_tensors,
                  std::vector<bm_tensor_t> &out_tensors);
  void net_launch_decode(int block_idx, int kv_offset,
                         bm_device_mem_t &input_mem, const int *position_id,
                         std::vector<uint16_t> &attention_mask);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
                  int size = 0);
  void init_tensors(const bm_net_info_t *net,
                    std::vector<bm_tensor_t> &in_tensors,
                    std::vector<bm_tensor_t> &out_tensors, int stage = 0);
  void init_by_names();

public:
  int token_length;
  int history_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int HIDDEN_SIZE;
  int KV_BYTES; // kv bytes for one token
  int NUM_LAYERS;
  int VIT_DIMS;
  int MAX_PATCHES;
  int MAX_PIXELS;
  int GATE_FRAMES; // T: number of video frames per gate decision (static)
  bool lmhead_with_topk;
  bool is_dynamic;
  bool prefill_mask;
  uint16_t mask_value;
  std::vector<int> visited_tokens;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_vit;
  const bm_net_info_t *net_gate;   // StreamMind Gate (PreNet+Mamba+PostNet)
  const bm_net_info_t *net_cls;   // ClsNet (4-layer Qwen3 binary classifier)
  bm_device_mem_t dev_buffer;
  bm_device_mem_t gate_buf;  // [1, T, 2560] bf16 - gate/cls shared workspace
  bm_device_mem_t cls_buf;   // [1, T, 2] bf16 - cls_net logits output
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void Mage_VL::init_tensors(const bm_net_info_t *net,
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

void Mage_VL::net_launch(const bm_net_info_t *net,
                         const std::vector<bm_tensor_t> &in_tensors,
                         std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
}

void Mage_VL::net_launch_decode(int idx, int kv_offset,
                                 bm_device_mem_t &input_mem,
                                 const int *pos_id,
                                 std::vector<uint16_t> &attention_mask) {
  auto &net = net_blocks_cache[idx];
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net, in_tensors, out_tensors);

  in_tensors[0].device_mem = input_mem;
  if (idx == 0) {
    bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem, (void *)pos_id);
    bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                  (void *)attention_mask.data());
  } else {
    // position_ids and attention_mask are identical across layers, so reuse the
    // mem filled by layer 0.
    in_tensors[1].device_mem = net_blocks_cache[0]->stages[0].input_mems[1];
    in_tensors[2].device_mem = net_blocks_cache[0]->stages[0].input_mems[2];
  }
  out_tensors[1].device_mem = bm_mem_from_device(
      past_key[idx].u.device.device_addr + kv_offset, KV_BYTES);
  out_tensors[2].device_mem = bm_mem_from_device(
      past_value[idx].u.device.device_addr + kv_offset, KV_BYTES);

  net_launch(net, in_tensors, out_tensors);
}

void Mage_VL::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                  int size) {
  if (!size) {
    size = bm_mem_get_device_size(src);
  }
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Mage_VL::clear_history() {
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
}

void Mage_VL::init_by_names() {
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
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);

  // Gate and ClsNet are optional — present only in streaming-enabled bmodels.
  bool has_gate = is_exist("gate", net_names, num_nets);
  bool has_cls = is_exist("cls_net", net_names, num_nets);
  if (has_gate && has_cls) {
    net_gate = bmrt_get_network_info(p_bmrt, "gate");
    net_cls = bmrt_get_network_info(p_bmrt, "cls_net");
  } else {
    net_gate = nullptr;
    net_cls = nullptr;
    if (has_gate != has_cls) {
      printf("Warning: gate (%s) and cls_net (%s) not both present; "
             "streaming disabled.\n",
             has_gate ? "found" : "missing",
             has_cls ? "found" : "missing");
    }
  }

  // Fixed nets: embedding, embedding_cache, vit, lm_head, [gate, cls_net].
  int fixed_nets = 4 + (has_gate ? 1 : 0) + (has_cls ? 1 : 0);
  auto num_blocks = num_nets - fixed_nets;
  NUM_LAYERS = num_blocks / 2; // block_ + block_cache_ for each layer
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    if ((!is_exist(block_name.c_str(), net_names, num_nets)) ||
        (!is_exist(cache_name.c_str(), net_names, num_nets))) {
      NUM_LAYERS = i;
      printf("Warning: Only %d blocks found, total %d blocks.\n", NUM_LAYERS,
             num_blocks);
      break;
    }
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
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
  is_dynamic = net_blocks[0]->is_dynamic;
  prefill_mask = net_blocks[0]->input_num > 2; // with prefill attention mask
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;
  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];
  MAX_PIXELS = MAX_PATCHES * 16 * 16;
  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  KV_BYTES =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  history_length = 0;
  printf("Num Layers:%d\n", NUM_LAYERS);
  printf("Max Pixels: %d (patches=%d)\n", MAX_PIXELS, MAX_PATCHES);

  if (net_gate) {
    GATE_FRAMES = net_gate->stages[0].input_shapes[0].dims[1];
    int gate_dim = net_gate->stages[0].input_shapes[0].dims[2];
    printf("Gate: T=%d, dim=%d\n", GATE_FRAMES, gate_dim);
    assert(gate_dim == HIDDEN_SIZE &&
           "gate hidden size must match LLM hidden size");
  } else {
    GATE_FRAMES = 0;
  }
}

void Mage_VL::init(int dev_id, std::string model_path) {
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

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  // dev_buffer holds the prefill embeddings (text + injected image).
  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);

  // Gate + ClsNet device buffers (only when streaming nets are present).
  if (net_gate) {
    int gate_bytes = GATE_FRAMES * HIDDEN_SIZE * sizeof(uint16_t);
    status = bm_malloc_device_byte(bm_handle, &gate_buf, gate_bytes);
    assert(BM_SUCCESS == status);
    int cls_bytes = GATE_FRAMES * 2 * sizeof(uint16_t);
    status = bm_malloc_device_byte(bm_handle, &cls_buf, cls_bytes);
    assert(BM_SUCCESS == status);
  }
}

void Mage_VL::deinit() {
  if (!p_bmrt)
    return; // already deinitialized (guards against double-free from __del__)
  bm_free_device(bm_handle, dev_buffer);
  if (net_gate) {
    bm_free_device(bm_handle, gate_buf);
    bm_free_device(bm_handle, cls_buf);
  }
  bmrt_destroy(p_bmrt);
  p_bmrt = nullptr;
  bm_dev_free(bm_handle);
}

void Mage_VL::forward_embed(ArrayInt const &tokens) {
  token_length = tokens.size();
  assert(token_length < SEQLEN);
  assert(token_length <= MAX_INPUT_LENGTH);

  auto p_buffer = tokens.request();
  auto p_tokens = static_cast<int *>(p_buffer.ptr);
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(p_tokens, p_tokens + token_length, visited_tokens.data());

  empty(bm_handle, dev_buffer);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed, in_tensors, out_tensors);
  if (token_length != MAX_INPUT_LENGTH) {
    empty(bm_handle, in_tensors[0].device_mem);
  }
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)visited_tokens.data(),
                        token_length * sizeof(int));
  net_launch(net_embed, in_tensors, out_tensors);
  d2d(dev_buffer, out_tensors[0].device_mem, 0,
      token_length * HIDDEN_SIZE * sizeof(uint16_t));
}

void Mage_VL::forward_vit(ArrayFloat const &pixel_values, ArrayInt const &pos_t,
                          ArrayInt const &pos_h, ArrayInt const &pos_w,
                          int vit_offset) {
  auto p_pix = pixel_values.request();
  auto p_pt = pos_t.request();
  auto p_ph = pos_h.request();
  auto p_pw = pos_w.request();
  int num_patches = p_pix.shape[0]; // == MAX_PATCHES for phase 1
  assert(num_patches == MAX_PATCHES);
  assert(pos_t.size() == num_patches);
  assert(pos_h.size() == num_patches);
  assert(pos_w.size() == num_patches);

  // bidirectional attention among the real patches: 0 in the top-left
  // num_patches x num_patches block, -1e4 elsewhere.
  std::vector<float> attention_mask(MAX_PATCHES * MAX_PATCHES, -10000.0f);
  for (int i = 0; i < num_patches; i++) {
    auto row = attention_mask.begin() + i * MAX_PATCHES;
    std::fill(row, row + num_patches, 0.0f);
  }

  empty_net(bm_handle, net_vit);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_vit, in_tensors, out_tensors);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)p_pix.ptr,
                        num_patches * VIT_DIMS * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                        (void *)p_pt.ptr, num_patches * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                        (void *)p_ph.ptr, num_patches * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[3].device_mem,
                        (void *)p_pw.ptr, num_patches * sizeof(int));
  bm_memcpy_s2d(bm_handle, in_tensors[4].device_mem,
                (void *)attention_mask.data());

  net_launch(net_vit, in_tensors, out_tensors);

  // inject merged image embeddings [num_patches/4, HIDDEN_SIZE] into dev_buffer
  // at the image_pad span (which starts right after the vision_start token).
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = (num_patches / 4) * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                     out_tensors[0].device_mem, 0, vit_size);
}

int Mage_VL::forward_first(ArrayInt const &position_ids) {
  // 1D position_ids (plain Qwen3 RoPE). Pad to MAX_INPUT_LENGTH.
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  std::vector<int> position_ids_pad(MAX_INPUT_LENGTH, 0);
  std::copy(p_ids, p_ids + token_length, position_ids_pad.begin());

  // causal mask (static): 0 for j <= i within [0, token_length), mask_value
  // elsewhere. Diagonal included so each token attends to itself.
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH,
                                       mask_value);
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j <= i; j++) {
      attention_mask[i * MAX_INPUT_LENGTH + j] = 0;
    }
  }

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors);
    in_tensors[0].device_mem = out_mem;
    if (idx == 0) {
      // position_ids and attention_mask are shared across layers; fill once.
      bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                    (void *)position_ids_pad.data());
      if (prefill_mask) {
        bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                      (void *)attention_mask.data());
      }
    }
    net_launch(net_blocks[idx], in_tensors, out_tensors);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1], 0,
        token_length * KV_BYTES);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2], 0,
        token_length * KV_BYTES);
  }

  // forward lm_head on the last token's hidden state. lm_head already does
  // the argmax internally (output [1,1] int32 == token_id).
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = bm_mem_from_device(
      out_mem.u.device.device_addr + (token_length - 1) * bytes, bytes);
  // lm_head output is [1,1] int32 (built-in argmax). Leave out_tensors[0]
  // pointing at net_lm's own output_mem (4 bytes); do NOT alias it to
  // dev_buffer, otherwise the full-size bm_memcpy_d2s below would copy the
  // entire 5 MB dev_buffer into a 4-byte stack int and overflow the stack.
  net_launch(net_lm, in_tensors, out_tensors);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_tensors[0].device_mem);
  visited_tokens[token_length] = token;
  token_length++;
  history_length = token_length;
  return token;
}

int Mage_VL::forward_next() {
  int real_len = SEQLEN;
  std::vector<uint16_t> attention_mask(real_len + 1, 0);
  for (int i = history_length - 1; i < real_len; i++) {
    attention_mask[i] = mask_value;
  }
  int32_t position_id = history_length - 1;

  // embedding_cache: single token id -> hidden state
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed_cache, in_tensors, out_tensors);
  int token = visited_tokens[token_length - 1];
  bm_memcpy_s2d(bm_handle, in_tensors[0].device_mem, (void *)&token);
  net_launch(net_embed_cache, in_tensors, out_tensors);
  auto out_mem = out_tensors[0].device_mem;

  // blocks
  int token_offset = (history_length - 1) * KV_BYTES;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    net_launch_decode(idx, token_offset, out_mem, &position_id, attention_mask);
    out_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
  }

  // forward lm_head
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = out_mem;
  net_launch(net_lm, in_tensors, out_tensors);

  token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_tensors[0].device_mem);
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

py::array_t<float> Mage_VL::read_vit_embeddings(int offset, int num_tokens) {
  assert(offset + num_tokens <= MAX_INPUT_LENGTH);
  int bytes = num_tokens * HIDDEN_SIZE * sizeof(uint16_t);
  // Create a sub-range view of dev_buffer and copy to host.
  auto sub_mem = bm_mem_from_device(
      dev_buffer.u.device.device_addr +
          offset * HIDDEN_SIZE * sizeof(uint16_t),
      bytes);
  std::vector<uint16_t> bf16_buf(num_tokens * HIDDEN_SIZE);
  bm_memcpy_d2s(bm_handle, bf16_buf.data(), sub_mem);
  // Convert bf16 -> f32 for Python.
  py::array_t<float> result({num_tokens, HIDDEN_SIZE});
  auto buf = result.mutable_unchecked<2>();
  for (int i = 0; i < num_tokens; i++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      buf(i, j) = bf16_to_f32(bf16_buf[i * HIDDEN_SIZE + j]);
    }
  }
  return result;
}

std::vector<float> Mage_VL::forward_gate(ArrayFloat const &averaged_tokens) {
  assert(net_gate && "gate net not available in this bmodel");
  auto buf = averaged_tokens.request();
  assert(buf.ndim == 2);
  int T = buf.shape[0];
  int D = buf.shape[1];
  assert(T == GATE_FRAMES);
  assert(D == HIDDEN_SIZE);
  auto *src = static_cast<float *>(buf.ptr);

  // Convert f32 -> bf16 and upload to gate_buf.
  std::vector<uint16_t> bf16_buf(T * D);
  for (int i = 0; i < T * D; i++) {
    uint32_t bits;
    memcpy(&bits, &src[i], sizeof(uint32_t));
    bf16_buf[i] = static_cast<uint16_t>(bits >> 16);
  }
  bm_memcpy_s2d(bm_handle, gate_buf, bf16_buf.data());

  // Launch gate net: [1, T, 2560] bf16 -> [1, T, 2560] bf16.
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_gate, in_tensors, out_tensors);
  in_tensors[0].device_mem = gate_buf;
  net_launch(net_gate, in_tensors, out_tensors);

  // Feed gate output directly into cls_net (same device address space).
  auto gate_out_mem = net_gate->stages[0].output_mems[0];
  init_tensors(net_cls, in_tensors, out_tensors);
  in_tensors[0].device_mem = gate_out_mem;
  out_tensors[0].device_mem = cls_buf;
  net_launch(net_cls, in_tensors, out_tensors);

  // Copy logits [1, T, 2] bf16 back to host.
  int cls_elems = T * 2;
  std::vector<uint16_t> logits_bf16(cls_elems);
  bm_memcpy_d2s(bm_handle, logits_bf16.data(), cls_buf);

  // Convert bf16 -> f32 and return as flat vector [T*2].
  std::vector<float> logits(cls_elems);
  for (int i = 0; i < cls_elems; i++) {
    logits[i] = bf16_to_f32(logits_bf16[i]);
  }
  return logits;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Mage_VL>(m, "Mage_VL")
      .def(pybind11::init<>())
      .def("init", &Mage_VL::init)
      .def("forward_embed", &Mage_VL::forward_embed)
      .def("forward_vit", &Mage_VL::forward_vit)
      .def("forward_first", &Mage_VL::forward_first)
      .def("forward_next", &Mage_VL::forward_next)
      .def("clear_history", &Mage_VL::clear_history)
      .def("forward_gate", &Mage_VL::forward_gate)
      .def("read_vit_embeddings", &Mage_VL::read_vit_embeddings)
      .def("deinit", &Mage_VL::deinit)
      .def_readonly("SEQLEN", &Mage_VL::SEQLEN)
      .def_readonly("MAX_PIXELS", &Mage_VL::MAX_PIXELS)
      .def_readonly("MAX_PATCHES", &Mage_VL::MAX_PATCHES)
      .def_readonly("MAX_INPUT_LENGTH", &Mage_VL::MAX_INPUT_LENGTH)
      .def_readonly("GATE_FRAMES", &Mage_VL::GATE_FRAMES)
      .def_readonly("token_length", &Mage_VL::token_length)
      .def_readonly("history_length", &Mage_VL::history_length);
}
