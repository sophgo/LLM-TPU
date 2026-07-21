//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// Falcon-Perception (early-fusion segmentation VLM) inference wrapper.
//
// bmodel net contract (f32, static, 256x256, seq512/max_input384):
//   embedding        : [1,384] i32            -> [1,384,1024] f32
//   embedding_cache  : [1,1]   i32            -> [1,1,1024]   f32
//   lm_head          : [1,1024] f32           -> [1,1] i32 (greedy topk)
//   block_{0..27}    : [1,384,1024] f32,
//                      [1,384] i32 (pos),
//                      [1,384,16,32] f32 (golden_cos),
//                      [1,384,16,32] f32 (golden_sin),
//                      [1,1,384,384] f32 (attn_mask)
//                    -> [1,384,1024] f32 (ffn_res, final block applies norm),
//                       [1,384,16,128] f32 (k), [1,384,16,128] f32 (v)
//   block_cache_{i}  : [1,1,1024] f32,
//                      [1,1] i32, [1,1,16,32] f32 x2, [1,1,1,513] f32 (mask),
//                      [1,512,16,128] f32 (history_k), [1,512,16,128] f32 (history_v)
//                    -> [1,1,1024] f32, [1,1,16,128] f32 (k), [1,1,16,128] f32 (v)
//   (k/v cached at 16 heads: HF repeats n_kv_heads=8 to n_heads=16 BEFORE the
//    2D golden RoPE, so the post-RoPE k cannot be compressed to 8 heads.)
//   coord_head/size_head : [1,1024] f32 -> [1,2,1024] f32
//   seg_head         : [1,1024] f32 -> [1,256] f32
//   mask_head        : [1,256,256,256] f32, [16,256] f32 -> [16,256,256] f32
//   coord_encoder/size_encoder : [1,2] f32 -> [1,1024] f32 (Fourier 回灌)
//   anyup            : [1,3,256,256] f32, [1,1024,16,16] f32 -> [1,256,256,256] f32
//                      (window_mask baked into bmodel as a const weight)
//
// Image-patch projection is done host-side (pipeline); forward_first embeds
// the tokens on device, then scatters the projected patch features directly
// into block_0's input mem at the img-token rows (partial s2d by offset) — the
// full [MAX_INPUT_LENGTH, HIDDEN] embedding never leaves the device. coord/size
// Fourier 回灌 is done via coord_encoder/size_encoder bmodels whose output
// overrides the embedded token in forward_next (fourier_emb arg).

#include "bmruntime_interface.h"
#include "memory.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

class FalconPerception {
public:
  void init(int devid, std::string model_path);
  void deinit();
  ArrayFloat forward_embed(ArrayInt const &tokens);
  // tokens: token ids [MAX_INPUT_LENGTH] (right-padded). img_feats/img_pos:
  // projected image-patch features [M,HIDDEN] and their token row positions [M]
  // (the img-token slots). forward_first embeds tokens on device, scatters
  // img_feats into block_0's input mem at img_pos rows, then runs prefill —
  // the full embedding never round-trips through the host.
  int forward_first(ArrayInt const &tokens, ArrayInt const &position_ids,
                    ArrayFloat const &golden_cos, ArrayFloat const &golden_sin,
                    ArrayFloat const &attention_mask, int token_length,
                    ArrayFloat const &img_feats, ArrayInt const &img_pos);
  // fourier_emb: if non-empty [1,HIDDEN] embedding, overrides the embedded token
  // (回灌: coord/size token's embedding replaced by Fourier(xy/hw)).
  int forward_next(int prev_token, ArrayInt const &position_ids,
                   ArrayFloat const &golden_cos,
                   ArrayFloat const &golden_sin,
                   ArrayFloat const &attention_mask,
                   ArrayFloat const &fourier_emb);
  // full prefill hidden [MAX_INPUT_LENGTH, HIDDEN] (block_27 post-norm output),
  // cached by forward_first before lm_head runs. Used to gather image-token
  // features for anyup and to read h at the last position.
  ArrayFloat forward_first_hidden();
  // h at the last (new) position [1, HIDDEN] — the lm_head input mem, valid right
  // after forward_first or forward_next. Heads (coord/size/seg) read this.
  ArrayFloat forward_hidden();
  ArrayFloat forward_coord(ArrayFloat const &hidden);
  ArrayFloat forward_size(ArrayFloat const &hidden);
  ArrayFloat forward_seg(ArrayFloat const &hidden);
  ArrayFloat forward_mask(ArrayFloat const &hr_features,
                           ArrayFloat const &segm_tokens);
  ArrayFloat forward_anyup(ArrayFloat const &images,
                            ArrayFloat const &lr_tokens);
  // Fourier encoders for 回灌: [1,2] (x,y)/(h,w) -> [1,HIDDEN]
  ArrayFloat forward_fourier_coord(ArrayFloat const &coords);
  ArrayFloat forward_fourier_size(ArrayFloat const &coords);
  void clear_history();

public:
  int token_length;
  int history_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int HIDDEN_SIZE;
  int KV_BYTES;
  int NUM_LAYERS;
  bool lmhead_with_topk;
  bool support_history;

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void init_by_names();

  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_coord, *net_size, *net_seg, *net_mask, *net_anyup;
  const bm_net_info_t *net_coord_enc, *net_size_enc;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
  // block_27 prefill hidden, d2s'd in forward_first BEFORE lm_head runs (lm_head's
  // launch reuses/clobbers block_27's output mem on some runtimes; reading it
  // afterward yields garbage). forward_first_hidden returns this cache.
  std::vector<float> prefill_hidden;
};

void FalconPerception::net_launch(const bm_net_info_t *net, int stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);
  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[stage_idx].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[stage_idx].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i],
                            net->stages[stage_idx].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[stage_idx].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  // bmrt reuses device mem globally, so a net's output mem can be aliased by a
  // later net's input mem and overwritten while still in flight. The natural
  // d2s at each forward_* tail does not cover these mid-sequence handoffs
  // (e.g. embed_cache -> fourier 回灌 s2d, lm -> embed_cache d2d), which race
  // across the GDMA/TPU engines and produce nondeterministic output. Syncing
  // after every launch is the safe fix; these nets are small and called few
  // times, so the cost is modest. The prefill block loop has its own per-layer
  // sync (see forward_first) for the same reason.
  bm_thread_sync(bm_handle);
}

void FalconPerception::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void FalconPerception::clear_history() {
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
}

void FalconPerception::init_by_names() {
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
  net_coord = bmrt_get_network_info(p_bmrt, "coord_head");
  net_size = bmrt_get_network_info(p_bmrt, "size_head");
  net_seg = bmrt_get_network_info(p_bmrt, "seg_head");
  net_mask = bmrt_get_network_info(p_bmrt, "mask_head");
  net_anyup = bmrt_get_network_info(p_bmrt, "anyup");
  net_coord_enc = bmrt_get_network_info(p_bmrt, "coord_encoder");
  net_size_enc = bmrt_get_network_info(p_bmrt, "size_encoder");

  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);
  // fixed nets: embed + embed_cache + lm_head + 4 heads + anyup + 2 fourier encoders = 10
  int num_fixed = 10;
  auto num_blocks = num_nets - num_fixed;
  NUM_LAYERS = num_blocks / 2;
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    if ((!is_exist(block_name.c_str(), net_names, num_nets)) ||
        (!is_exist(cache_name.c_str(), net_names, num_nets))) {
      NUM_LAYERS = i;
      printf("Warning: Only %d blocks found.\n", NUM_LAYERS);
      break;
    }
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }
  free(net_names);

  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;
  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[5].dims[1];
  KV_BYTES =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  support_history = true;
  history_length = 0;
  printf("Num Layers:%d  SEQLEN:%d  MAX_INPUT_LENGTH:%d  HIDDEN:%d\n",
         NUM_LAYERS, SEQLEN, MAX_INPUT_LENGTH, HIDDEN_SIZE);
}

void FalconPerception::init(int dev_id, std::string model_path) {
  std::cout << "Device [ " << dev_id << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  assert(BM_SUCCESS == status);
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");
  print_devmem_info(bm_handle);
  init_by_names();
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[5];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[6];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
}

void FalconPerception::deinit() {
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

// Embed a batch of tokens (left-padded to MAX_INPUT_LENGTH by caller).
// Returns the token embedding [MAX_INPUT_LENGTH, HIDDEN_SIZE] as a numpy array
// so the pipeline can scatter image-patch features host-side before prefill.
ArrayFloat FalconPerception::forward_embed(ArrayInt const &tokens) {
  std::vector<int> input_ids(MAX_INPUT_LENGTH, 0);
  auto num = tokens.size();
  auto p_buffer = tokens.request();
  auto p_tokens = static_cast<int *>(p_buffer.ptr);
  std::copy(p_tokens, p_tokens + num, input_ids.data());
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)input_ids.data());
  net_launch(net_embed);
  ArrayFloat out({MAX_INPUT_LENGTH, HIDDEN_SIZE});
  bm_memcpy_d2s(bm_handle, out.mutable_data(), out_mem);
  token_length = num;
  return out;
}

int FalconPerception::forward_first(ArrayInt const &tokens,
                                    ArrayInt const &position_ids,
                                    ArrayFloat const &golden_cos,
                                    ArrayFloat const &golden_sin,
                                    ArrayFloat const &attention_mask,
                                    int tok_len,
                                    ArrayFloat const &img_feats,
                                    ArrayInt const &img_pos) {
  token_length = tok_len;
  auto p_tok = tokens.request();
  auto p_pos = position_ids.request();
  auto p_gcos = golden_cos.request();
  auto p_gsin = golden_sin.request();
  auto p_mask = attention_mask.request();
  auto p_feats = img_feats.request();
  auto p_ipos = img_pos.request();

  // 1) token embedding on device (no host round-trip). The projected image-
  //    patch features are scattered directly into block_0's input mem at the
  //    img-token rows below, so the full [MAX_INPUT_LENGTH, HIDDEN] embedding
  //    never leaves the device.
  auto &embed_in = net_embed->stages[0].input_mems[0];
  auto &embed_out = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, embed_in, p_tok.ptr);
  net_launch(net_embed);

  // 2) place embedding into block_0's input mem (d2d). embed_out and block_0's
  //    input mem may alias under BM_RUNTIME_SHARE_MEM (then this is a harmless
  //    self-copy). net_launch above synced, so embed_out is settled.
  auto &blk0_in = net_blocks[0]->stages[0].input_mems[0];
  int embed_bytes = MAX_INPUT_LENGTH * HIDDEN_SIZE * sizeof(float);
  bm_memcpy_d2d_byte(bm_handle, blk0_in, 0, embed_out, 0, embed_bytes);

  // 3) scatter img-patch features into the img-token rows (partial s2d by row
  //    offset). Overwrites the placeholder img-token embeddings with the
  //    projected patch features, matching the old host-side masked_scatter.
  int M = p_ipos.size;
  int row_bytes = HIDDEN_SIZE * sizeof(float);
  auto ipos_ptr = static_cast<int *>(p_ipos.ptr);
  auto feats_ptr = static_cast<float *>(p_feats.ptr);
  for (int m = 0; m < M; m++) {
    int row = ipos_ptr[m];
    auto dst = bm_mem_from_device(
        blk0_in.u.device.device_addr + (uint64_t)row * row_bytes, row_bytes);
    bm_memcpy_s2d(bm_handle, dst,
                  (void *)(feats_ptr + (int64_t)m * HIDDEN_SIZE));
  }

  // 4) run prefill blocks
  bm_device_mem_t out_mem;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &net = net_blocks[idx];
    std::vector<bm_tensor_t> in_tensors(net->input_num);
    std::vector<bm_tensor_t> out_tensors(net->output_num);
    // block_0 input already filled (embedding + img scatter) above; later
    // blocks read the previous block's output.
    if (idx > 0) {
      bm_memcpy_d2d_byte(bm_handle, net->stages[0].input_mems[0], 0, out_mem, 0,
                         token_length * HIDDEN_SIZE * sizeof(float));
    }
    // shared runtime inputs (pos/gcos/gsin/mask): s2d into THIS block's own input
    // mems every iteration. bmrt reuses device memory globally, so block_0's
    // input mems get overwritten by later blocks' outputs — reusing block_0's
    // mems (the old approach) made block_1+ read garbage pos/gcos/gsin/mask.
    bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[1], p_pos.ptr);
    bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[2], p_gcos.ptr);
    bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[3], p_gsin.ptr);
    bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[4], p_mask.ptr);
    for (int i = 0; i < net->input_num; i++) {
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
    out_mem = net->stages[0].output_mems[0];
    d2d(past_key[idx], net->stages[0].output_mems[1]);
    d2d(past_value[idx], net->stages[0].output_mems[2]);
    // bmrt reuses device mem globally: this block's input_mems[1..4] can alias
    // the previous block's output region. The per-iteration s2d of pos/gcos/
    // gsin/mask above is a GDMA write that would race the prior TPU launch on
    // the same mem without a barrier. Sync here so the next iteration's s2d
    // sees a settled device state (prefill runs once per query, cheap).
    bm_thread_sync(bm_handle);
  }

  // lm_head on the last real token's hidden (block_27 already applied norm)
  // Capture block_27's full hidden to host BEFORE lm_head: its launch can reuse
  // (clobber) block_27's output mem, so a later d2s (forward_first_hidden) would
  // read garbage. The block loop's trailing bm_thread_sync settled out_mem.
  prefill_hidden.assign(MAX_INPUT_LENGTH * HIDDEN_SIZE, 0.0f);
  bm_memcpy_d2s(bm_handle, prefill_hidden.data(), out_mem);
  int bytes = HIDDEN_SIZE * sizeof(float);
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * bytes, bytes);
  net_launch(net_lm);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  token_length++;
  history_length = token_length;
  return token;
}

int FalconPerception::forward_next(int prev_token,
                                   ArrayInt const &position_ids,
                                   ArrayFloat const &golden_cos,
                                   ArrayFloat const &golden_sin,
                                   ArrayFloat const &attention_mask,
                                   ArrayFloat const &fourier_emb) {
  auto p_pos = position_ids.request();
  auto p_gcos = golden_cos.request();
  auto p_gsin = golden_sin.request();
  auto p_mask = attention_mask.request();

  // embed the previous token. lm_head's output mem (token_id) gets reused/clobbered
  // by later nets (anyup, heads) on some runtimes, so d2d from it yields garbage.
  // The caller already holds prev_token (the lm_head argmax), so s2d it directly.
  auto &ec_in = net_embed_cache->stages[0].input_mems[0];
  auto &ec_out = net_embed_cache->stages[0].output_mems[0];
  int32_t tok = (int32_t)prev_token;
  bm_memcpy_s2d(bm_handle, ec_in, &tok);
  net_launch(net_embed_cache);
  // 回灌: if a Fourier-encoded embedding is provided (coord/size token),
  // overwrite the embedded token before it enters block_cache_0. net_launch
  // syncs above, so this s2d into ec_out cannot race the embed_cache launch.
  if (fourier_emb.size() > 0) {
    bm_memcpy_s2d(bm_handle, ec_out, fourier_emb.request().ptr);
  }

  int token_offset = (history_length - 1) * KV_BYTES;
  bm_device_mem_t out_mem = ec_out;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &net = net_blocks_cache[idx];
    std::vector<bm_tensor_t> in_tensors(net->input_num);
    std::vector<bm_tensor_t> out_tensors(net->output_num);
    // input_states = embedding_cache output (first block) / prev block output
    bmrt_tensor_with_device(&in_tensors[0], out_mem, net->input_dtypes[0],
                            net->stages[0].input_shapes[0]);
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[1], p_pos.ptr);
      bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[2], p_gcos.ptr);
      bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[3], p_gsin.ptr);
      bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[4], p_mask.ptr);
    }
    for (int i = 1; i < 5; i++) {
      bmrt_tensor_with_device(&in_tensors[i],
                              net_blocks_cache[0]->stages[0].input_mems[i],
                              net->input_dtypes[i],
                              net->stages[0].input_shapes[i]);
    }
    bmrt_tensor_with_device(&in_tensors[5], past_key[idx], net->input_dtypes[5],
                            net->stages[0].input_shapes[5]);
    bmrt_tensor_with_device(&in_tensors[6], past_value[idx],
                            net->input_dtypes[6], net->stages[0].input_shapes[6]);
    for (int i = 0; i < net->output_num; i++) {
      bmrt_tensor_with_device(&out_tensors[i], net->stages[0].output_mems[i],
                              net->output_dtypes[i],
                              net->stages[0].output_shapes[i]);
    }
    // write new k/v into the kv cache at the current token offset
    auto k_mem = bm_mem_from_device(
        past_key[idx].u.device.device_addr + token_offset, KV_BYTES);
    auto v_mem = bm_mem_from_device(
        past_value[idx].u.device.device_addr + token_offset, KV_BYTES);
    bmrt_tensor_with_device(&out_tensors[1], k_mem, net->output_dtypes[1],
                            net->stages[0].output_shapes[1]);
    bmrt_tensor_with_device(&out_tensors[2], v_mem, net->output_dtypes[2],
                            net->stages[0].output_shapes[2]);
    auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                     net->input_num, out_tensors.data(),
                                     net->output_num, true, false);
    assert(ret);
    // symmetric with forward_first's prefill loop: consecutive block_cache
    // launches chain via out_mem (block idx+1 reads block idx output_mems[0]
    // directly). bmrt reuses device mem globally, so without a barrier the
    // next iteration's reads/writes can race the prior launch on aliased mem.
    bm_thread_sync(bm_handle);
    out_mem = net->stages[0].output_mems[0];
  }

  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  token_length++;
  history_length++;
  return token;
}

// Full prefill hidden states [MAX_INPUT_LENGTH, HIDDEN] from block_27 output.
// forward_first already d2s'd block_27's output into prefill_hidden BEFORE
// lm_head ran (lm_head's launch reuses/clobbers that device mem on some
// runtimes); this just returns the cache. Call right after forward_first.
ArrayFloat FalconPerception::forward_first_hidden() {
  ArrayFloat out({MAX_INPUT_LENGTH, HIDDEN_SIZE});
  std::copy(prefill_hidden.begin(), prefill_hidden.end(), out.mutable_data());
  return out;
}

// h at the last (new) position [1, HIDDEN] — the lm_head input mem, filled by
// forward_first (last token slice) and forward_next (d2d from block_cache out).
ArrayFloat FalconPerception::forward_hidden() {
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  ArrayFloat out({1, HIDDEN_SIZE});
  bm_memcpy_d2s(bm_handle, out.mutable_data(), lm_in_mem);
  return out;
}

// ---- Fourier encoders for 回灌: [1,2] -> [1, HIDDEN] ----
ArrayFloat FalconPerception::forward_fourier_coord(ArrayFloat const &coords) {
  auto &net = net_coord_enc;
  bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[0], coords.request().ptr);
  net_launch(net);
  auto &out_mem = net->stages[0].output_mems[0];
  ArrayFloat out({1, HIDDEN_SIZE});
  bm_memcpy_d2s(bm_handle, out.mutable_data(), out_mem);
  return out;
}

ArrayFloat FalconPerception::forward_fourier_size(ArrayFloat const &coords) {
  auto &net = net_size_enc;
  bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[0], coords.request().ptr);
  net_launch(net);
  auto &out_mem = net->stages[0].output_mems[0];
  ArrayFloat out({1, HIDDEN_SIZE});
  bm_memcpy_d2s(bm_handle, out.mutable_data(), out_mem);
  return out;
}

// ---- heads (segmentation: coord/size/seg/mask) ----
ArrayFloat FalconPerception::forward_coord(ArrayFloat const &hidden) {
  auto &net = net_coord;
  auto p = hidden.request();
  bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[0], p.ptr);
  net_launch(net);
  auto &out_mem = net->stages[0].output_mems[0];
  int n = net->stages[0].output_shapes[0].dims[1] *
          net->stages[0].output_shapes[0].dims[2];
  ArrayFloat out({1, 2, n / 2});
  bm_memcpy_d2s(bm_handle, out.mutable_data(), out_mem);
  return out;
}

ArrayFloat FalconPerception::forward_size(ArrayFloat const &hidden) {
  auto &net = net_size;
  auto p = hidden.request();
  bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[0], p.ptr);
  net_launch(net);
  auto &out_mem = net->stages[0].output_mems[0];
  int n = net->stages[0].output_shapes[0].dims[1] *
          net->stages[0].output_shapes[0].dims[2];
  ArrayFloat out({1, 2, n / 2});
  bm_memcpy_d2s(bm_handle, out.mutable_data(), out_mem);
  return out;
}

ArrayFloat FalconPerception::forward_seg(ArrayFloat const &hidden) {
  auto &net = net_seg;
  auto p = hidden.request();
  bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[0], p.ptr);
  net_launch(net);
  auto &out_mem = net->stages[0].output_mems[0];
  int n = net->stages[0].output_shapes[0].dims[1];
  ArrayFloat out({1, n});
  bm_memcpy_d2s(bm_handle, out.mutable_data(), out_mem);
  return out;
}

ArrayFloat FalconPerception::forward_mask(ArrayFloat const &hr_features,
                                          ArrayFloat const &segm_tokens) {
  auto &net = net_mask;
  auto p_hr = hr_features.request();
  auto p_sg = segm_tokens.request();
  bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[0], p_hr.ptr);
  bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[1], p_sg.ptr);
  net_launch(net);
  auto &out_mem = net->stages[0].output_mems[0];
  auto &shp = net->stages[0].output_shapes[0];
  ArrayFloat out({shp.dims[0], shp.dims[1], shp.dims[2]});
  bm_memcpy_d2s(bm_handle, out.mutable_data(), out_mem);
  return out;
}

ArrayFloat FalconPerception::forward_anyup(ArrayFloat const &images,
                                           ArrayFloat const &lr_tokens) {
  auto &net = net_anyup;
  bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[0],
                images.request().ptr);
  bm_memcpy_s2d(bm_handle, net->stages[0].input_mems[1],
                lr_tokens.request().ptr);
  net_launch(net);
  auto &out_mem = net->stages[0].output_mems[0];
  auto &shp = net->stages[0].output_shapes[0];
  ArrayFloat out({shp.dims[0], shp.dims[1], shp.dims[2], shp.dims[3]});
  bm_memcpy_d2s(bm_handle, out.mutable_data(), out_mem);
  return out;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<FalconPerception>(m, "FalconPerception")
      .def(pybind11::init<>())
      .def("init", &FalconPerception::init)
      .def("deinit", &FalconPerception::deinit)
      .def("forward_embed", &FalconPerception::forward_embed)
      .def("forward_first", &FalconPerception::forward_first,
           py::arg("tokens"), py::arg("position_ids"), py::arg("golden_cos"),
           py::arg("golden_sin"), py::arg("attention_mask"),
           py::arg("token_length"), py::arg("img_feats"),
           py::arg("img_pos"))
      .def("forward_next", &FalconPerception::forward_next,
           py::arg("prev_token"),
           py::arg("position_ids"), py::arg("golden_cos"), py::arg("golden_sin"),
           py::arg("attention_mask"),
           py::arg("fourier_emb") = ArrayFloat())
      .def("forward_first_hidden", &FalconPerception::forward_first_hidden)
      .def("forward_hidden", &FalconPerception::forward_hidden)
      .def("forward_coord", &FalconPerception::forward_coord)
      .def("forward_size", &FalconPerception::forward_size)
      .def("forward_seg", &FalconPerception::forward_seg)
      .def("forward_mask", &FalconPerception::forward_mask)
      .def("forward_anyup", &FalconPerception::forward_anyup)
      .def("forward_fourier_coord", &FalconPerception::forward_fourier_coord)
      .def("forward_fourier_size", &FalconPerception::forward_fourier_size)
      .def("clear_history", &FalconPerception::clear_history)
      .def_readonly("SEQLEN", &FalconPerception::SEQLEN)
      .def_readonly("MAX_INPUT_LENGTH", &FalconPerception::MAX_INPUT_LENGTH)
      .def_readonly("HIDDEN_SIZE", &FalconPerception::HIDDEN_SIZE)
      .def_readonly("NUM_LAYERS", &FalconPerception::NUM_LAYERS)
      .def_readonly("support_history", &FalconPerception::support_history)
      .def_readonly("history_length", &FalconPerception::history_length);
}
