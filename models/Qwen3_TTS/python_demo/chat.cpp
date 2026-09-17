//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// Qwen3-TTS pipeline (full TPU, strategy B). One bmodel, 100 nets:
//   - Talker (28-layer Qwen3 LM, MRoPE interleaved [24,20,20], QK-norm):
//       block_0..27 (prefill), block_cache_0..27 (decode), embedding /
//       embedding_cache (text_embedding + text_projection MLP), codec_embedding /
//       codec_embedding_cache (3072x1024 lookup), lm_head (= codec_head, raw
//       logits [1,3072]) + greedy_head + sample_head (compiled with
//       --do_sample; sample_head natively does repetition_penalty + top_k +
//       top_p + temperature on TPU; host does the multinomial draw).
//   - CodePredictor (5-layer, 1D RoPE): cp_block_cache_0..4 + 15 cp_embed_j +
//       15 cp_lm_head_j. Each audio frame runs 1 Talker decode step + 16 CP
//       steps -> 16 codec codes (code0 from Talker, code1..15 from CP).
//   - speaker_encoder (ECAPA-TDNN): mel [1,SPK_MEL_MAX,128] (STATIC) -> spk_embed
//       [1,1024]. The pipeline zero-pads the full-length reference mel to
//       SPK_MEL_MAX frames (default 750 ~= 8s; no 300-frame truncation).
//   - mimi_decoder: codec codes [1,16,T] -> wav [T*1920] (24 kHz). Handles
//       any T by chunking internally to 256-frame blocks.
//   - Multi-language: codec_prefill supports language_id via codec_embedding,
//       think/nothink mode control via CODEC_THINK / CODEC_NOTHINK tokens.
//   - Streaming: forward_mimi_decoder supports streaming by processing in
//       chunks; the pipeline writes wav incrementally during generation.
//
// Scope: x-vector-only voice clone (reference wav -> speaker embedding),
// streaming trailing (text_body + tts_eos, not non-streaming tts_pad), multi-language
// via codec_language_id, think/nothink mode control, streaming audio out (chunked
// mimi_decoder), fully static shapes. In-context-learning (ref_code) clone mode
// is not supported (the mimi_encoder net is compiled only when
// QWEN3_TTS_ENABLE_ICL is set; pipeline.py raises NotImplementedError on
// --ref_text). The forward_mimi_encoder path below is preserved but unused.
//
// Mask convention (verified against tpu-mlir gen_block_cache_by_length):
// block_cache concatenates [history_k (SEQLEN slots), new_k (1)] and runs
// FAttention over SEQLEN+1 positions; the new token sits at the LAST mask
// index (SEQLEN). The decode mask therefore keeps index SEQLEN unmasked (self
// attention) and masks the garbage history slots [kv_filled, SEQLEN-1]. With
// history_length = kv_filled + 1, that is: mask_value for i in
// [history_length-1, SEQLEN), attend [0, history_length-2] U {SEQLEN}.
// =============================================================================

#include "bmruntime_interface.h"
#include "memory.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <numeric>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdexcept>
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

static inline float bf16_to_f32(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(float));
  return f;
}

static inline uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(uint32_t));
  return static_cast<uint16_t>(bits >> 16);  // truncate (round-to-zero)
}

void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
  (void)ret;
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

class Qwen3TTS {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void clear_history();

  // ---- primitive embed / encoder / decoder nets (host-visible results) ----
  // speaker_encoder: mel [SPK_MEL_MAX,128] f32 (static; SPK_MEL_MAX = --audio_length, default 750 ~= 8s) -> spk [1024]
  py::array_t<float> forward_speaker_encoder(ArrayFloat const &mel);
  // embedding net (text_embedding + text_projection MLP): tokens -> [n,1024] f32
  py::array_t<float> text_embedding(ArrayInt const &tokens);
  // codec_embedding net (3072x1024 lookup): tokens -> [n,1024] f32
  py::array_t<float> codec_embedding(ArrayInt const &tokens);
  // CP embed net j (2048x1024 lookup, code group j+1): tokens -> [n,1024] f32.
  // Batch lookup by launching the single-token net n times (cheap Gather net).
  py::array_t<float> cp_embed_group(ArrayInt const &tokens, int group);
  // write the assembled prefill embeds [seq,1024] f32 into dev_buffer[0:seq]
  void set_talker_prefill(ArrayFloat const &embeds);
  // write trailing_text_hidden [SEQLEN,1024] f32 (padded with tts_pad by caller)
  void set_trailing(ArrayFloat const &trailing);

  // Talker prefill over dev_buffer -> first code0. Position_ids are computed
  // internally (pure-text MRoPE: all 3 axes = [0..P-1]).
  int forward_first();
  // Talker decode one frame: run 16-step CodePredictor, assemble sum-of-16
  // codec embeds + trailing[gen_step], run 28 block_cache -> next code0.
  // Position_ids computed internally (all 3 axes = history_length - 1).
  int forward_next();
  // mimi_decoder: codes [16,T] int32 -> wav [T*1920] f32 (chunked to 256 frames)
  py::array_t<float> forward_mimi_decoder(ArrayInt const &codes);
  // mimi_encoder: wav [T_audio] f32 (24kHz, zero-padded to ENC_T_AUDIO) ->
  // codes [16, T_12] int32 (12.5Hz). Causal -> trailing pad is exact; caller
  // slices to real T_12 = real_samples / 1920. Used for ICL voice clone.
  py::array_t<int> forward_mimi_encoder(ArrayFloat const &wav);

  // Sampling config (HF defaults). code0 uses the TPU sample_head net (RP +
  // top_k/top_p/temperature on device, multinomial on host); CP code1..15 use
  // host-side top_k/top_p/temperature sampling (no RP — HF does not pass RP to
  // the CodePredictor). do_sample=false falls back to greedy argmax for both.
  void set_sampling(float temperature, int top_k, float top_p,
                    float repetition_penalty, bool do_sample,
                    float sub_temperature, int sub_top_k, float sub_top_p,
                    int repetition_window, int64_t seed);

  std::vector<int> get_frame_codes() { return last_frame_codes; }

public:
  int token_length;     // total Talker tokens (prefill + generated)
  int history_length;   // = kv_filled + 1 (Qwen3 convention)
  int gen_step;         // decode step index for trailing_text_hidden
  int SEQLEN;           // 2048
  int MAX_INPUT_LENGTH; // 1024
  int HIDDEN_SIZE;      // 1024
  int KV_BYTES;         // per-token Talker K/V bytes
  int NUM_LAYERS;       // 28
  int CP_NUM_LAYERS;    // 5
  int CP_NUM_CODE_GROUPS; // 16
  int CP_HIDDEN_SIZE;  // 1024
  int CP_KV_BYTES;     // per-token CP K/V bytes
  int CP_SEQLEN;        // CP KV history slots (2x num_code_groups; CP KV is
                        // fresh per frame, never reaches Talker's SEQLEN)
  int MIMI_FRAME;      // 256 (static codec frames per mimi_decoder call)
  int MIMI_HOP;        // 1920 (audio samples per codec frame, 24kHz/12.5Hz)
  int SPK_MEL_MAX;     // 750 default (speaker_encoder static mel frames, runtime-read from net; = --audio_length, ~8s)
  int ENC_T_AUDIO;     // 192000 default (mimi_encoder static wav samples, ~8s @24kHz = SPK_MEL_MAX*256; runtime-read from net)
  int ENC_T_12;        // 256 (static mimi_encoder output codec frames @12.5Hz)
  int ENC_N_Q;         // 16 (mimi_encoder output codebooks)
  int ENC_DOWNSAMPLE;  // 1920 (wav samples per 12.5Hz codec frame)
  int CODEC_VOCAB;     // 3072 (Talker codec_head / sample_head vocab)
  int CP_VOCAB;        // 2048 (CP lm_head vocab)
  uint16_t mask_value;
  std::vector<int> visited_tokens;

  // sampling config
  float temperature = 0.9f;
  int top_k = 50;
  float top_p = 1.0f;
  float repetition_penalty = 1.05f;
  float sub_temperature = 0.9f;
  int sub_top_k = 50;
  float sub_top_p = 1.0f;
  int repetition_window = 1024;
  bool do_sample = true;
  std::mt19937 sgen;
  std::vector<int> gen_code0_seq;  // generated code0 history (for Talker RP)

private:
  void net_launch(const bm_net_info_t *net,
                  const std::vector<bm_tensor_t> &in_tensors,
                  std::vector<bm_tensor_t> &out_tensors);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int dst_off = 0,
                  int size = 0);
  void init_tensors(const bm_net_info_t *net,
                    std::vector<bm_tensor_t> &in_tensors,
                    std::vector<bm_tensor_t> &out_tensors, int stage = 0);
  int cp_argmax(bm_device_mem_t &logits_mem);  // d2s + argmax over 2048
  int cp_sample(bm_device_mem_t &logits_mem);  // host top_k/top_p/temp sampling
  int sample_code0(bm_device_mem_t &logits_mem);  // TPU sample_head or greedy
  void launch_cp_decode(int layer, bm_device_mem_t &input_mem, int cp_pos,
                        int cp_kv_filled, int token_offset);
  void run_code_predictor(bm_device_mem_t &talker_past_hidden, int code0);

  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;       // 28
  std::vector<const bm_net_info_t *> net_blocks_cache; // 28
  std::vector<const bm_net_info_t *> net_cp_blocks_cache; // 5
  std::vector<const bm_net_info_t *> net_cp_embed;      // 15
  std::vector<const bm_net_info_t *> net_cp_lm_head;    // 15
  const bm_net_info_t *net_embedding;
  const bm_net_info_t *net_embedding_cache;
  const bm_net_info_t *net_codec_embedding;
  const bm_net_info_t *net_codec_embedding_cache;
  const bm_net_info_t *net_lm;          // codec_head -> raw logits [1,3072]
  const bm_net_info_t *net_greedy_head; // TopK K=1 over 3072 (greedy fallback)
  const bm_net_info_t *net_sample_head; // RP+top_k+top_p+temp on TPU (6in/2out)
  const bm_net_info_t *net_speaker_encoder;
  const bm_net_info_t *net_mimi_decoder;
  const bm_net_info_t *net_mimi_encoder = nullptr;  // ICL; absent in older bmodels

  bm_device_mem_t dev_buffer;            // [MAX_INPUT_LENGTH, HIDDEN] bf16
  bm_device_mem_t trailing_mem;          // [SEQLEN, HIDDEN] bf16
  bm_device_mem_t talker_past_hidden_mem; // [1, HIDDEN] bf16
  bm_device_mem_t decode_input_mem;       // [1,1,HIDDEN] bf16 (assembled step input)
  bm_device_mem_t cp_hidden_mem;          // [1,1,CP_HIDDEN] bf16 (CP single hidden)
  bm_device_mem_t cp_two_token_mem;       // [1,2,CP_HIDDEN] bf16 (CP prefill cat)
  bm_device_mem_t mimi_input_mem;         // [1,16,MIMI_FRAME] int32
  bm_device_mem_t mimi_output_mem;        // [1,1,MIMI_FRAME*MIMI_HOP] bf16
  bm_device_mem_t enc_input_mem;          // [1,1,ENC_T_AUDIO] f32
  bm_device_mem_t enc_output_mem;         // [ENC_N_Q,ENC_T_12] f32 (TopK indices cast to f32 by lowering; values are exact integers 0..2047)
  std::vector<bm_device_mem_t> past_key;    // 28
  std::vector<bm_device_mem_t> past_value;  // 28
  std::vector<bm_device_mem_t> cp_past_key;   // 5
  std::vector<bm_device_mem_t> cp_past_value; // 5
  std::vector<int> last_frame_codes;  // 16
};

void Qwen3TTS::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int dst_off,
                   int size) {
  if (size == 0) {
    size = bm_mem_get_device_size(src);
  }
  auto ret = bm_memcpy_d2d_byte(bm_handle, dst, dst_off, src, 0, size);
  assert(BM_SUCCESS == ret);
}

void Qwen3TTS::init_tensors(const bm_net_info_t *net,
                            std::vector<bm_tensor_t> &in_tensors,
                            std::vector<bm_tensor_t> &out_tensors,
                            int stage) {
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

void Qwen3TTS::net_launch(const bm_net_info_t *net,
                          const std::vector<bm_tensor_t> &in_tensors,
                          std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  if (!ret) {
    std::cerr << "[net_launch FAIL] " << net->name << " in_num=" << net->input_num
              << " out_num=" << net->output_num << " in_shapes:";
    for (int i = 0; i < net->input_num; i++) {
      std::cerr << " [";
      for (int d = 0; d < in_tensors[i].shape.num_dims; d++)
        std::cerr << in_tensors[i].shape.dims[d]
                  << (d + 1 < in_tensors[i].shape.num_dims ? "," : "");
      std::cerr << "]";
    }
    std::cerr << std::endl;
    // Throw instead of abort(): pybind11 converts this to a Python exception so
    // the process can call deinit() and exit cleanly. An abort() mid-inference
    // previously left the TPU A53 subsystem in a wedged state that even
    // bm-smi -recovery could not clear (required a host reboot).
    throw std::runtime_error(std::string("net_launch failed: ") + net->name);
  }
  bm_thread_sync(bm_handle);
}

void Qwen3TTS::init(int dev_id, std::string model_path) {
  std::cout << "Device [ " << dev_id << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  if (status != BM_SUCCESS)
    throw std::runtime_error("bm_dev_request failed");
  p_bmrt = bmrt_create(bm_handle);
  if (p_bmrt == NULL)
    throw std::runtime_error("bmrt_create failed");
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  if (!ret) {
    std::cerr << "[ERROR] bmrt_load_bmodel failed for " << model_path << "\n";
    print_devmem_info(bm_handle);
    throw std::runtime_error("bmrt_load_bmodel failed");
  }
  bm_thread_sync(bm_handle);
  printf("Done!\n");
  print_devmem_info(bm_handle);

  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0)
        return true;
    }
    return false;
  };
  net_embedding = bmrt_get_network_info(p_bmrt, "embedding");
  net_embedding_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_codec_embedding = bmrt_get_network_info(p_bmrt, "codec_embedding");
  net_codec_embedding_cache =
      bmrt_get_network_info(p_bmrt, "codec_embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  net_speaker_encoder = bmrt_get_network_info(p_bmrt, "speaker_encoder");
  net_mimi_decoder = bmrt_get_network_info(p_bmrt, "mimi_decoder");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);

  // sample_head / greedy_head are present when the bmodel was compiled with
  // --do_sample (codec_head -> raw logits). Fall back to greedy argmax on host
  // if absent (e.g. an older static bmodel).
  net_sample_head = nullptr;
  net_greedy_head = nullptr;
  if (is_exist("sample_head", net_names, num_nets))
    net_sample_head = bmrt_get_network_info(p_bmrt, "sample_head");
  if (is_exist("greedy_head", net_names, num_nets))
    net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  // mimi_encoder (ICL voice clone) is absent in older bmodels.
  if (is_exist("mimi_encoder", net_names, num_nets))
    net_mimi_encoder = bmrt_get_network_info(p_bmrt, "mimi_encoder");

  NUM_LAYERS = 0;
  for (int i = 0;; i++) {
    std::string b = "block_" + std::to_string(i);
    std::string c = "block_cache_" + std::to_string(i);
    if (!is_exist(b.c_str(), net_names, num_nets) ||
        !is_exist(c.c_str(), net_names, num_nets))
      break;
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, b.c_str()));
    net_blocks_cache.emplace_back(bmrt_get_network_info(p_bmrt, c.c_str()));
    NUM_LAYERS++;
  }
  CP_NUM_LAYERS = 0;
  for (int i = 0;; i++) {
    std::string c = "cp_block_cache_" + std::to_string(i);
    if (!is_exist(c.c_str(), net_names, num_nets))
      break;
    net_cp_blocks_cache.emplace_back(bmrt_get_network_info(p_bmrt, c.c_str()));
    CP_NUM_LAYERS++;
  }
  CP_NUM_CODE_GROUPS = 0;
  for (int i = 0;; i++) {
    std::string e = "cp_embed_" + std::to_string(i);
    std::string l = "cp_lm_head_" + std::to_string(i);
    if (!is_exist(e.c_str(), net_names, num_nets) ||
        !is_exist(l.c_str(), net_names, num_nets))
      break;
    net_cp_embed.emplace_back(bmrt_get_network_info(p_bmrt, e.c_str()));
    net_cp_lm_head.emplace_back(bmrt_get_network_info(p_bmrt, l.c_str()));
    CP_NUM_CODE_GROUPS++;
  }
  free(net_names);

  // CP_NUM_CODE_GROUPS counts code1..15 (15). code0 from Talker -> 16 total.
  assert(CP_NUM_CODE_GROUPS == 15);

  if (net_embedding_cache->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2;
  } else if (net_embedding_cache->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C;
  } else {
    throw std::runtime_error("Invalid attention dtype");
  }
  MAX_INPUT_LENGTH = net_embedding->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  KV_BYTES = bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  CP_HIDDEN_SIZE = net_cp_lm_head[0]->stages[0].input_shapes[0].dims[1];
  CP_KV_BYTES =
      bm_mem_get_device_size(net_cp_blocks_cache[0]->stages[0].output_mems[1]);
  // CP KV history depth, read from the cp_block_cache net (compiled at
  // 2x num_code_groups, NOT SEQLEN — CP KV is fresh per frame).
  CP_SEQLEN = net_cp_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  MIMI_FRAME = net_mimi_decoder->stages[0].input_shapes[0].dims[2];
  MIMI_HOP = net_mimi_decoder->stages[0].output_shapes[0].dims[2] / MIMI_FRAME;
  // codec_head (lm_head) output vocab = 3072; CP lm_head vocab = 2048.
  CODEC_VOCAB = net_lm->stages[0].output_shapes[0].dims[1];
  CP_VOCAB = net_cp_lm_head[0]->stages[0].output_shapes[0].dims[1];
  // speaker_encoder static mel frames (input dim 1 = SPK_MEL_MAX, default 750).
  SPK_MEL_MAX = net_speaker_encoder->stages[0].input_shapes[0].dims[1];
  // mimi_encoder static shapes (only if present). Input [1,1,T_audio] f32,
  // output [N_Q, T_12] int32. ENC_DOWNSAMPLE = T_audio / T_12 (= 1920).
  if (net_mimi_encoder) {
    ENC_T_AUDIO = net_mimi_encoder->stages[0].input_shapes[0].dims[2];
    ENC_N_Q = net_mimi_encoder->stages[0].output_shapes[0].dims[0];
    ENC_T_12 = net_mimi_encoder->stages[0].output_shapes[0].dims[1];
    ENC_DOWNSAMPLE = ENC_T_AUDIO / ENC_T_12;
  } else {
    ENC_T_AUDIO = ENC_T_12 = ENC_N_Q = ENC_DOWNSAMPLE = 0;
  }
  printf("Talker layers: %d, CP layers: %d, CP code groups-1: %d\n",
         NUM_LAYERS, CP_NUM_LAYERS, CP_NUM_CODE_GROUPS);
  printf("SEQLEN: %d, CP_SEQLEN: %d, mimi: %dx%d\n", SEQLEN, CP_SEQLEN, MIMI_FRAME, MIMI_HOP);
  printf("codec_vocab: %d, cp_vocab: %d, spk_mel_max: %d\n",
         CODEC_VOCAB, CP_VOCAB, SPK_MEL_MAX);
  printf("sample_head: %s\n", net_sample_head ? "yes" : "no(greedy)");
  printf("mimi_encoder: %s\n", net_mimi_encoder ? "yes" : "no(no ICL)");

  visited_tokens.resize(SEQLEN);
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  // Talker block_cache nets also share the same input_mems[3]/[4] (bmrt reuses
  // mem for identical-shape inputs), so we CANNOT alias past_key[i] to
  // input_mems[3] — all 28 layers would collide on one buffer. Allocate an
  // independent KV cache per layer instead.
  for (int i = 0; i < NUM_LAYERS; i++) {
    status = bm_malloc_device_byte(bm_handle, &past_key[i],
        SEQLEN * KV_BYTES);  // KV_BYTES = 8*128*2 (per-token)
    assert(BM_SUCCESS == status);
    status = bm_malloc_device_byte(bm_handle, &past_value[i],
        SEQLEN * KV_BYTES);
    assert(BM_SUCCESS == status);
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  cp_past_key.resize(CP_NUM_LAYERS);
  cp_past_value.resize(CP_NUM_LAYERS);
  // CP block_cache nets share the same input_mems[3]/[4] (bmrt reuses mem for
  // identical-shape inputs), so we CANNOT alias past_key[i] to input_mems[3]
  // — all 5 layers would collide on one buffer, each overwriting the previous
  // layer's KV. Allocate an independent KV cache per layer instead.
  // size = [1, CP_SEQLEN, 8, 128] bf16 = CP_SEQLEN * 8 * 128 * 2 bytes.
  // CP_SEQLEN << SEQLEN (CP KV is fresh per frame, depth = num_code_groups).
  for (int i = 0; i < CP_NUM_LAYERS; i++) {
    status = bm_malloc_device_byte(bm_handle, &cp_past_key[i],
        CP_SEQLEN * CP_KV_BYTES);  // CP_KV_BYTES = 8*128*2 (per-token), xCP_SEQLEN slots
    assert(BM_SUCCESS == status);
    status = bm_malloc_device_byte(bm_handle, &cp_past_value[i],
        CP_SEQLEN * CP_KV_BYTES);
    assert(BM_SUCCESS == status);
    empty(bm_handle, cp_past_key[i]);
    empty(bm_handle, cp_past_value[i]);
  }
  // dev_buffer holds the prefill embeds [MAX_INPUT_LENGTH, HIDDEN] bf16
  status = bm_malloc_device_byte(bm_handle, &dev_buffer,
                                 MAX_INPUT_LENGTH * HIDDEN_SIZE * sizeof(uint16_t));
  assert(BM_SUCCESS == status);
  status = bm_malloc_device_byte(bm_handle, &trailing_mem,
                                 SEQLEN * HIDDEN_SIZE * sizeof(uint16_t));
  assert(BM_SUCCESS == status);
  status = bm_malloc_device_byte(bm_handle, &talker_past_hidden_mem,
                                 HIDDEN_SIZE * sizeof(uint16_t));
  assert(BM_SUCCESS == status);
  status = bm_malloc_device_byte(bm_handle, &decode_input_mem,
                                 HIDDEN_SIZE * sizeof(uint16_t));
  assert(BM_SUCCESS == status);
  status = bm_malloc_device_byte(bm_handle, &cp_hidden_mem,
                                 CP_HIDDEN_SIZE * sizeof(uint16_t));
  assert(BM_SUCCESS == status);
  status = bm_malloc_device_byte(bm_handle, &cp_two_token_mem,
                                 2 * CP_HIDDEN_SIZE * sizeof(uint16_t));
  assert(BM_SUCCESS == status);
  // mimi io (static [1,16,MIMI_FRAME] -> [1,1,MIMI_FRAME*MIMI_HOP])
  status = bm_malloc_device_byte(
      bm_handle, &mimi_input_mem,
      1 * 16 * MIMI_FRAME * sizeof(int32_t));
  assert(BM_SUCCESS == status);
  status = bm_malloc_device_byte(
      bm_handle, &mimi_output_mem,
      1 * 1 * MIMI_FRAME * MIMI_HOP * sizeof(uint16_t));
  assert(BM_SUCCESS == status);
  // mimi_encoder io (static [1,1,ENC_T_AUDIO] f32 -> [ENC_N_Q,ENC_T_12] f32)
  if (net_mimi_encoder) {
    status = bm_malloc_device_byte(bm_handle, &enc_input_mem,
                                   1 * 1 * ENC_T_AUDIO * sizeof(float));
    assert(BM_SUCCESS == status);
    status = bm_malloc_device_byte(bm_handle, &enc_output_mem,
                                   ENC_N_Q * ENC_T_12 * sizeof(float));
    assert(BM_SUCCESS == status);
  }
  token_length = 0;
  history_length = 0;
  gen_step = 0;
  last_frame_codes.resize(16, 0);
}

void Qwen3TTS::deinit() {
  // free Talker KV caches
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_free_device(bm_handle, past_key[i]);
    bm_free_device(bm_handle, past_value[i]);
  }
  // free CP KV caches
  for (int i = 0; i < CP_NUM_LAYERS; i++) {
    bm_free_device(bm_handle, cp_past_key[i]);
    bm_free_device(bm_handle, cp_past_value[i]);
  }
  bm_free_device(bm_handle, dev_buffer);
  bm_free_device(bm_handle, trailing_mem);
  bm_free_device(bm_handle, talker_past_hidden_mem);
  bm_free_device(bm_handle, decode_input_mem);
  bm_free_device(bm_handle, cp_hidden_mem);
  bm_free_device(bm_handle, cp_two_token_mem);
  bm_free_device(bm_handle, mimi_input_mem);
  bm_free_device(bm_handle, mimi_output_mem);
  if (net_mimi_encoder) {
    bm_free_device(bm_handle, enc_input_mem);
    bm_free_device(bm_handle, enc_output_mem);
  }
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void Qwen3TTS::clear_history() {
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  for (int i = 0; i < CP_NUM_LAYERS; i++) {
    empty(bm_handle, cp_past_key[i]);
    empty(bm_handle, cp_past_value[i]);
  }
  history_length = 0;
  token_length = 0;
  gen_step = 0;
  gen_code0_seq.clear();
}

int Qwen3TTS::cp_argmax(bm_device_mem_t &logits_mem) {
  // cp_lm_head outputs logits [1, CP_VOCAB] bf16; argmax on host.
  std::vector<uint16_t> bf16(CP_VOCAB);
  bm_memcpy_d2s(bm_handle, bf16.data(), logits_mem);
  int argmax = 0;
  float maxv = bf16_to_f32(bf16[0]);
  for (int i = 1; i < CP_VOCAB; i++) {
    float v = bf16_to_f32(bf16[i]);
    if (v > maxv) {
      maxv = v;
      argmax = i;
    }
  }
  return argmax;
}

int Qwen3TTS::cp_sample(bm_device_mem_t &logits_mem) {
  // Host-side sampling for CP code1..15 (vocab CP_VOCAB=2048). HF does not
  // pass repetition_penalty to the CodePredictor, so this applies only
  // top_k + top_p + temperature. Falls back to greedy argmax when !do_sample.
  if (!do_sample)
    return cp_argmax(logits_mem);
  std::vector<uint16_t> bf16(CP_VOCAB);
  bm_memcpy_d2s(bm_handle, bf16.data(), logits_mem);
  std::vector<float> logits(CP_VOCAB);
  for (int i = 0; i < CP_VOCAB; i++)
    logits[i] = bf16_to_f32(bf16[i]);
  int k = std::min(sub_top_k, CP_VOCAB);
  std::vector<int> idx(CP_VOCAB);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
  // softmax(logits[TopK] / temperature)
  std::vector<float> probs(k);
  float maxl = logits[idx[0]];
  for (int i = 0; i < k; i++)
    probs[i] = expf((logits[idx[i]] - maxl) / sub_temperature);
  float sum = 0.0f;
  for (int i = 0; i < k; i++)
    sum += probs[i];
  // top_p (nucleus) filter: keep the smallest prefix whose cumulative prob
  // reaches sub_top_p.
  int keep = k;
  float cum = 0.0f;
  for (int i = 0; i < k; i++) {
    cum += probs[i] / sum;
    if (cum >= sub_top_p) {
      keep = i + 1;
      break;
    }
  }
  std::vector<float> p(keep);
  for (int i = 0; i < keep; i++)
    p[i] = probs[i];
  std::discrete_distribution<> dist(p.begin(), p.end());
  return idx[dist(sgen)];
}

int Qwen3TTS::sample_code0(bm_device_mem_t &logits_mem) {
  // Talker code0 (codec vocab CODEC_VOCAB=3072). logits_mem is the lm_head
  // output [1, CODEC_VOCAB] f32. When do_sample and net_sample_head are
  // available, the TPU sample_head net does repetition_penalty + top_k +
  // top_p + temperature; the host only draws from the returned top-k
  // distribution. Otherwise fall back to greedy (greedy_head net or host
  // argmax).
  std::vector<bm_tensor_t> in, out;
  if (do_sample && net_sample_head) {
    init_tensors(net_sample_head, in, out);
    d2d(in[0].device_mem, logits_mem, 0, CODEC_VOCAB * sizeof(float));
    // past GENERATED code0 ids for repetition penalty (sliding window). The
    // first generated code0 has an empty history -> no RP. The Talker prefill
    // is embeds (not codec ids), so it is excluded (matches HF generate with
    // inputs_embeds, where RP only sees generated tokens).
    int penalty_len = (int)gen_code0_seq.size();
    const int *penalty_ptr = gen_code0_seq.data();
    if (repetition_window > 0 && penalty_len > repetition_window) {
      penalty_len = repetition_window;
      penalty_ptr = gen_code0_seq.data() + (gen_code0_seq.size() - repetition_window);
    }
    // The sample_head net requires a non-zero-length input_ids (0-length
    // GatherElements is invalid). With no generated history, feed a single
    // dummy id and penalty=1.0 so RP is a no-op (penalizing the dummy by 1.0
    // leaves all logits unchanged).
    int dummy_id = 0;
    float rp_eff = repetition_penalty;
    if (penalty_len == 0) {
      penalty_len = 1;
      penalty_ptr = &dummy_id;
      rp_eff = 1.0f;
    }
    bm_memcpy_s2d_partial(bm_handle, in[1].device_mem, (void *)penalty_ptr,
                          penalty_len * sizeof(int));
    in[1].shape.dims[1] = penalty_len;
    bm_memcpy_s2d(bm_handle, in[2].device_mem, (void *)&rp_eff);
    bm_memcpy_s2d(bm_handle, in[3].device_mem, (void *)&temperature);
    bm_memcpy_s2d(bm_handle, in[4].device_mem, (void *)&top_k);
    bm_memcpy_s2d(bm_handle, in[5].device_mem, (void *)&top_p);
    net_launch(net_sample_head, in, out);
    int cand = top_k;
    std::vector<float> probs(cand);
    bm_memcpy_d2s_partial_offset(bm_handle, probs.data(), out[0].device_mem,
                                 cand * sizeof(float), 0);
    std::vector<int> tokens(cand);
    bm_memcpy_d2s_partial_offset(bm_handle, tokens.data(), out[1].device_mem,
                                 cand * sizeof(int), 0);
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return tokens[dist(sgen)];
  }
  if (net_greedy_head) {
    init_tensors(net_greedy_head, in, out);
    d2d(in[0].device_mem, logits_mem, 0, CODEC_VOCAB * sizeof(float));
    net_launch(net_greedy_head, in, out);
    int token = 0;
    bm_memcpy_d2s(bm_handle, (void *)&token, out[0].device_mem);
    return token;
  }
  // host argmax fallback (no sampling nets compiled in)
  std::vector<float> logits(CODEC_VOCAB);
  bm_memcpy_d2s(bm_handle, logits.data(), logits_mem);
  int argmax = 0;
  float maxv = logits[0];
  for (int i = 1; i < CODEC_VOCAB; i++)
    if (logits[i] > maxv) {
      maxv = logits[i];
      argmax = i;
    }
  return argmax;
}

void Qwen3TTS::set_sampling(float temperature_, int top_k_, float top_p_,
                            float repetition_penalty_, bool do_sample_,
                            float sub_temperature_, int sub_top_k_,
                            float sub_top_p_, int repetition_window_,
                            int64_t seed) {
  temperature = temperature_;
  top_k = top_k_;
  top_p = top_p_;
  repetition_penalty = repetition_penalty_;
  do_sample = do_sample_;
  sub_temperature = sub_temperature_;
  sub_top_k = sub_top_k_;
  sub_top_p = sub_top_p_;
  repetition_window = repetition_window_;
  sgen.seed(seed);
}

// ---- host-visible embed nets ----

py::array_t<float> Qwen3TTS::forward_speaker_encoder(ArrayFloat const &mel) {
  // mel: [SPK_MEL_MAX, 128] f32 (static; SPK_MEL_MAX default 750). The pipeline
  // zero-pads the full-length reference mel to SPK_MEL_MAX frames on the host
  // (no 300-frame truncation), so the embedding sees the whole reference audio. The net is
  // STATIC [1, SPK_MEL_MAX (default 750), 128] -> [1, 1024].
  auto p = mel.request();
  assert(p.ndim == 2 && p.shape[1] == 128);
  int T = p.shape[0];
  if (T != SPK_MEL_MAX) {
    throw std::runtime_error(
        "forward_speaker_encoder: mel T=" + std::to_string(T) +
        " != SPK_MEL_MAX=" + std::to_string(SPK_MEL_MAX) +
        " (pipeline must zero-pad the reference mel to SPK_MEL_MAX)");
  }
  auto *src = static_cast<float *>(p.ptr);
  std::vector<bm_tensor_t> in, out;
  init_tensors(net_speaker_encoder, in, out);
  // static net: input is already [1, SPK_MEL_MAX, 128], no runtime shape override.
  bm_memcpy_s2d_partial(bm_handle, in[0].device_mem, (void *)src,
                        SPK_MEL_MAX * 128 * sizeof(float));
  net_launch(net_speaker_encoder, in, out);
  // output [1, 1024] bf16 -> host f32
  std::vector<uint16_t> bf16(1024);
  bm_memcpy_d2s(bm_handle, bf16.data(), out[0].device_mem);
  py::array_t<float> result(1024);
  auto r = result.mutable_unchecked<1>();
  for (int i = 0; i < 1024; i++)
    r(i) = bf16_to_f32(bf16[i]);
  return result;
}

py::array_t<float> Qwen3TTS::text_embedding(ArrayInt const &tokens) {
  int n = tokens.size();
  assert(n <= MAX_INPUT_LENGTH);
  auto p = tokens.request();
  auto *src = static_cast<int *>(p.ptr);
  std::vector<bm_tensor_t> in, out;
  init_tensors(net_embedding, in, out);
  empty(bm_handle, in[0].device_mem);
  bm_memcpy_s2d_partial(bm_handle, in[0].device_mem, (void *)src,
                        n * sizeof(int));
  net_launch(net_embedding, in, out);
  // output [1, MAX_INPUT_LENGTH, HIDDEN] bf16; take first n rows
  std::vector<uint16_t> bf16(n * HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, bf16.data(), out[0].device_mem,
                        n * HIDDEN_SIZE * sizeof(uint16_t));
  py::array_t<float> result({n, HIDDEN_SIZE});
  auto r = result.mutable_unchecked<2>();
  for (int i = 0; i < n * HIDDEN_SIZE; i++)
    r(i / HIDDEN_SIZE, i % HIDDEN_SIZE) = bf16_to_f32(bf16[i]);
  return result;
}

py::array_t<float> Qwen3TTS::codec_embedding(ArrayInt const &tokens) {
  int n = tokens.size();
  assert(n <= MAX_INPUT_LENGTH);
  auto p = tokens.request();
  auto *src = static_cast<int *>(p.ptr);
  std::vector<bm_tensor_t> in, out;
  init_tensors(net_codec_embedding, in, out);
  empty(bm_handle, in[0].device_mem);
  bm_memcpy_s2d_partial(bm_handle, in[0].device_mem, (void *)src,
                        n * sizeof(int));
  net_launch(net_codec_embedding, in, out);
  std::vector<uint16_t> bf16(n * HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, bf16.data(), out[0].device_mem,
                        n * HIDDEN_SIZE * sizeof(uint16_t));
  py::array_t<float> result({n, HIDDEN_SIZE});
  auto r = result.mutable_unchecked<2>();
  for (int i = 0; i < n * HIDDEN_SIZE; i++)
    r(i / HIDDEN_SIZE, i % HIDDEN_SIZE) = bf16_to_f32(bf16[i]);
  return result;
}

py::array_t<float> Qwen3TTS::cp_embed_group(ArrayInt const &tokens, int group) {
  // cp_embed_{group} net: single-token lookup [1,1] -> [1,1,HIDDEN]. Launch n
  // times for a batch of n codes (the nets are tiny Gather ops).
  assert(group >= 0 && group < CP_NUM_CODE_GROUPS);
  int n = tokens.size();
  auto p = tokens.request();
  auto *src = static_cast<int *>(p.ptr);
  std::vector<bm_tensor_t> in, out;
  init_tensors(net_cp_embed[group], in, out);
  std::vector<uint16_t> bf16(n * HIDDEN_SIZE);
  for (int i = 0; i < n; i++) {
    bm_memcpy_s2d(bm_handle, in[0].device_mem, (void *)&src[i]);
    net_launch(net_cp_embed[group], in, out);
    bm_memcpy_d2s_partial(bm_handle, bf16.data() + i * HIDDEN_SIZE, out[0].device_mem,
                          HIDDEN_SIZE * sizeof(uint16_t));
  }
  py::array_t<float> result({n, HIDDEN_SIZE});
  auto r = result.mutable_unchecked<2>();
  for (int i = 0; i < n * HIDDEN_SIZE; i++)
    r(i / HIDDEN_SIZE, i % HIDDEN_SIZE) = bf16_to_f32(bf16[i]);
  return result;
}

void Qwen3TTS::set_talker_prefill(ArrayFloat const &embeds) {
  // embeds: [seq, HIDDEN] f32 -> dev_buffer[0:seq] bf16
  auto p = embeds.request();
  assert(p.ndim == 2 && p.shape[1] == HIDDEN_SIZE);
  int seq = p.shape[0];
  assert(seq <= MAX_INPUT_LENGTH);
  auto *src = static_cast<float *>(p.ptr);
  empty(bm_handle, dev_buffer);  // zero padding region
  std::vector<uint16_t> bf16(seq * HIDDEN_SIZE);
  for (int i = 0; i < seq * HIDDEN_SIZE; i++)
    bf16[i] = f32_to_bf16(src[i]);
  bm_memcpy_s2d_partial(bm_handle, dev_buffer, (void *)bf16.data(),
                        seq * HIDDEN_SIZE * sizeof(uint16_t));
  token_length = seq;
  gen_step = 0;
}

void Qwen3TTS::set_trailing(ArrayFloat const &trailing) {
  // trailing: [SEQLEN, HIDDEN] f32 -> trailing_mem bf16
  auto p = trailing.request();
  assert(p.ndim == 2 && p.shape[0] == SEQLEN && p.shape[1] == HIDDEN_SIZE);
  auto *src = static_cast<float *>(p.ptr);
  std::vector<uint16_t> bf16(SEQLEN * HIDDEN_SIZE);
  for (int i = 0; i < SEQLEN * HIDDEN_SIZE; i++)
    bf16[i] = f32_to_bf16(src[i]);
  bm_memcpy_s2d(bm_handle, trailing_mem, (void *)bf16.data());
}

// ---- CP decode helper (1-token block_cache) ----
void Qwen3TTS::launch_cp_decode(int layer, bm_device_mem_t &input_mem,
                                int cp_pos, int cp_kv_filled,
                                int token_offset) {
  auto net = net_cp_blocks_cache[layer];
  std::vector<bm_tensor_t> in, out;
  init_tensors(net, in, out);
  in[0].device_mem = input_mem;
  if (layer == 0) {
    int32_t pos = cp_pos;
    bm_memcpy_s2d(bm_handle, in[1].device_mem, (void *)&pos);
    // mask [1,1,1,CP_SEQLEN+1]: attend [0, cp_kv_filled-1] U {CP_SEQLEN},
    // mask_value for [cp_kv_filled, CP_SEQLEN-1]. (history_length=cp_kv_filled+1)
    std::vector<uint16_t> mask(CP_SEQLEN + 1, 0);
    for (int i = cp_kv_filled; i < CP_SEQLEN; i++)
      mask[i] = mask_value;
    bm_memcpy_s2d(bm_handle, in[2].device_mem, (void *)mask.data());
  } else {
    in[1].device_mem = net_cp_blocks_cache[0]->stages[0].input_mems[1];
    in[2].device_mem = net_cp_blocks_cache[0]->stages[0].input_mems[2];
  }
  // history_k/v: point at the CP KV cache (cp_past_key/cp_past_value), NOT the
  // net's own pre-allocated input_mems. FAttention concatenates [history, new_k]
  // and attends; the new k/v is written at token_offset.
  in[3].device_mem = cp_past_key[layer];
  in[4].device_mem = cp_past_value[layer];
  out[1].device_mem = bm_mem_from_device(
      cp_past_key[layer].u.device.device_addr + token_offset, CP_KV_BYTES);
  out[2].device_mem = bm_mem_from_device(
      cp_past_value[layer].u.device.device_addr + token_offset, CP_KV_BYTES);
  net_launch(net, in, out);
}

void Qwen3TTS::run_code_predictor(bm_device_mem_t &talker_past_hidden,
                                  int code0) {
  // CP KV fresh each frame.
  for (int i = 0; i < CP_NUM_LAYERS; i++) {
    empty(bm_handle, cp_past_key[i]);
    empty(bm_handle, cp_past_value[i]);
  }
  std::vector<bm_tensor_t> in, out;

  // embed(code0) via codec_embedding_cache (Talker's 3072 table == last_id_hidden)
  init_tensors(net_codec_embedding_cache, in, out);
  int id0 = code0;
  bm_memcpy_s2d(bm_handle, in[0].device_mem, (void *)&id0);
  net_launch(net_codec_embedding_cache, in, out);
  bm_device_mem_t code0_embed = out[0].device_mem;  // [1,1,1024] bf16
  // The bmodel may assign codec_embedding_cache's output (code0_embed) and
  // cp_block_cache's output to the SAME device address (address-assignment
  // reuse for same-shape [1,1,1024] tensors). Step A then overwrites
  // code0_embed with the cp_block_cache hidden, so step B reads zero. Copy
  // code0_embed to an independently-allocated scratch (cp_hidden_mem) before
  // step A, and feed that to step B.
  int cp_bytes = CP_HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, cp_hidden_mem, 0, code0_embed, 0, cp_bytes);
  bm_device_mem_t code0_embed_safe = cp_hidden_mem;

  // Step A (prefill token 0 = past_hidden, pos 0, history empty)
  //   input = talker_past_hidden (1 token); writes KV[0].
  bm_device_mem_t cur_in = talker_past_hidden;
  for (int layer = 0; layer < CP_NUM_LAYERS; layer++) {
    launch_cp_decode(layer, cur_in, /*cp_pos*/ 0, /*cp_kv_filled*/ 0,
                      /*token_offset*/ 0);
    cur_in = net_cp_blocks_cache[layer]->stages[0].output_mems[0];
  }
  // Step B (prefill token 1 = code0_embed, pos 1, history=[KV[0]])
  //   hidden_B -> cp_lm_head[0] -> code1
  cur_in = code0_embed_safe;  // use the pre-step-A copy (original is clobbered)
  for (int layer = 0; layer < CP_NUM_LAYERS; layer++) {
    launch_cp_decode(layer, cur_in, /*cp_pos*/ 1, /*cp_kv_filled*/ 1,
                     /*token_offset*/ 1 * cp_bytes);
    cur_in = net_cp_blocks_cache[layer]->stages[0].output_mems[0];
  }
  bm_memcpy_d2d_byte(bm_handle, cp_hidden_mem, 0, cur_in, 0, cp_bytes);
  init_tensors(net_cp_lm_head[0], in, out);
  in[0].device_mem = cp_hidden_mem;
  net_launch(net_cp_lm_head[0], in, out);
  int code1 = cp_sample(out[0].device_mem);
  last_frame_codes[1] = code1;

  // Decode steps k=1..14: input = cp_embed[k-1](code_k), pos = k+1,
  //   history grows; cp_lm_head[k] -> code_{k+1}.
  int cp_kv_filled = 2;  // KV[0], KV[1] filled after steps A,B
  for (int k = 1; k < CP_NUM_CODE_GROUPS; k++) {  // k = 1..14
    int prev_code = last_frame_codes[k];            // code_k
    init_tensors(net_cp_embed[k - 1], in, out);
    bm_memcpy_s2d(bm_handle, in[0].device_mem, (void *)&prev_code);
    net_launch(net_cp_embed[k - 1], in, out);
    cur_in = out[0].device_mem;
    int cp_pos = k + 1;
    int token_offset = cp_kv_filled * cp_bytes;
    for (int layer = 0; layer < CP_NUM_LAYERS; layer++) {
      launch_cp_decode(layer, cur_in, cp_pos, cp_kv_filled, token_offset);
      cur_in = net_cp_blocks_cache[layer]->stages[0].output_mems[0];
    }
    bm_memcpy_d2d_byte(bm_handle, cp_hidden_mem, 0, cur_in, 0, cp_bytes);
    init_tensors(net_cp_lm_head[k], in, out);
    in[0].device_mem = cp_hidden_mem;
    net_launch(net_cp_lm_head[k], in, out);
    int code_next = cp_sample(out[0].device_mem);
    last_frame_codes[k + 1] = code_next;
    cp_kv_filled++;
  }
}

int Qwen3TTS::forward_first() {
  // Pure-text MRoPE: all 3 axes = [0..P-1]. Padded to [3, MAX_INPUT_LENGTH].
  int P = token_length;
  std::vector<int> pos_pad(3 * MAX_INPUT_LENGTH, 0);
  for (int a = 0; a < 3; a++)
    for (int i = 0; i < P; i++)
      pos_pad[a * MAX_INPUT_LENGTH + i] = i;

  // Causal mask [MAX_INPUT_LENGTH, MAX_INPUT_LENGTH]: 0 for j <= i within
  // [0, P), mask_value elsewhere. Padding rows (i >= P) are all masked
  // (their hidden/KV are discarded).
  std::vector<uint16_t> attn_mask(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH,
                                  mask_value);
  for (int i = 0; i < P; i++)
    for (int j = 0; j <= i; j++)
      attn_mask[i * MAX_INPUT_LENGTH + j] = 0;

  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in, out;
  bm_device_mem_t out_mem = dev_buffer;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    init_tensors(net_blocks[idx], in, out);
    in[0].device_mem = out_mem;  // read full [1, MAX, HIDDEN]
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in[1].device_mem, (void *)pos_pad.data());
      bm_memcpy_s2d(bm_handle, in[2].device_mem, (void *)attn_mask.data());
    }
    net_launch(net_blocks[idx], in, out);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx],
        const_cast<bm_device_mem_t &>(net_blocks[idx]->stages[0].output_mems[1]),
        0, P * KV_BYTES);
    d2d(past_value[idx],
        const_cast<bm_device_mem_t &>(net_blocks[idx]->stages[0].output_mems[2]),
        0, P * KV_BYTES);
  }

  // last hidden (position P-1) -> talker_past_hidden_mem
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, talker_past_hidden_mem, 0, out_mem,
                     (P - 1) * bytes, bytes);

  // codec_head on last hidden -> [1,3072] logits -> code0 (sample/greedy)
  init_tensors(net_lm, in, out);
  in[0].device_mem = talker_past_hidden_mem;
  net_launch(net_lm, in, out);
  int code0 = sample_code0(out[0].device_mem);
  last_frame_codes[0] = code0;
  visited_tokens[token_length] = code0;
  gen_code0_seq.push_back(code0);
  token_length++;        // P+1
  history_length = token_length;  // = P+1 = kv_filled(P) + 1
  gen_step = 0;
  return code0;
}

int Qwen3TTS::forward_next() {
  // Position_ids (MRoPE) computed internally = history_length - 1 (all 3 axes).
  int code0 = visited_tokens[token_length - 1];

  // 1. CodePredictor -> fill last_frame_codes[1..15] for THIS frame's code0.
  run_code_predictor(talker_past_hidden_mem, code0);
  // The current frame's 16 codes are [code0, code1..15]; expose via
  // get_frame_codes(). (last_frame_codes[0] is set here, NOT to next_code0.)
  last_frame_codes[0] = code0;

  // 2. Assemble Talker decode input = sum of 16 codec embeds + trailing[gen_step]
  //    e0 = codec_embed(code0); e_c = cp_embed_{c-1}(code_c) for c=1..15
  std::vector<float> acc(HIDDEN_SIZE, 0.0f);
  std::vector<bm_tensor_t> in, out;

  // e0: codec_embedding_cache(code0)
  init_tensors(net_codec_embedding_cache, in, out);
  bm_memcpy_s2d(bm_handle, in[0].device_mem, (void *)&code0);
  net_launch(net_codec_embedding_cache, in, out);
  std::vector<uint16_t> bf16(HIDDEN_SIZE);
  bm_memcpy_d2s(bm_handle, bf16.data(), out[0].device_mem);
  for (int j = 0; j < HIDDEN_SIZE; j++)
    acc[j] += bf16_to_f32(bf16[j]);

  // e_c (c=1..15): cp_embed_{c-1}(code_c)
  for (int c = 1; c < CP_NUM_CODE_GROUPS + 1; c++) {  // c = 1..15
    int code = last_frame_codes[c];
    int ei = c - 1;
    init_tensors(net_cp_embed[ei], in, out);
    bm_memcpy_s2d(bm_handle, in[0].device_mem, (void *)&code);
    net_launch(net_cp_embed[ei], in, out);
    bm_memcpy_d2s(bm_handle, bf16.data(), out[0].device_mem);
    for (int j = 0; j < HIDDEN_SIZE; j++)
      acc[j] += bf16_to_f32(bf16[j]);
  }

  // + trailing[gen_step] (clamp to SEQLEN-1)
  int tstep = std::min(gen_step, SEQLEN - 1);
  bm_device_mem_t trailing_slice = bm_mem_from_device(
      trailing_mem.u.device.device_addr + tstep * HIDDEN_SIZE * sizeof(uint16_t),
      HIDDEN_SIZE * sizeof(uint16_t));
  bm_memcpy_d2s(bm_handle, bf16.data(), trailing_slice);
  for (int j = 0; j < HIDDEN_SIZE; j++)
    acc[j] += bf16_to_f32(bf16[j]);

  // s2d assembled bf16 -> decode_input_mem
  for (int j = 0; j < HIDDEN_SIZE; j++)
    bf16[j] = f32_to_bf16(acc[j]);
  bm_memcpy_s2d(bm_handle, decode_input_mem, (void *)bf16.data());

  // 3. Talker decode: 28 block_cache. Position_ids (MRoPE) = history_length-1
  int kv_filled = history_length - 1;  // = P, P+1, ... (valid history length)
  std::vector<uint16_t> attn_mask(SEQLEN + 1, 0);
  for (int i = kv_filled; i < SEQLEN; i++)
    attn_mask[i] = mask_value;  // keep index SEQLEN (self)
  int32_t pos_id = history_length - 1;
  int32_t pos_id3[3] = {pos_id, pos_id, pos_id};
  int token_offset = kv_filled * KV_BYTES;

  bm_device_mem_t cur_in = decode_input_mem;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto net = net_blocks_cache[idx];
    init_tensors(net, in, out);
    in[0].device_mem = cur_in;
    // position_ids and attention_mask must be set for EVERY layer (not just idx==0),
    // otherwise the net reads uninitialized garbage memory.
    bm_memcpy_s2d(bm_handle, in[1].device_mem, (void *)pos_id3);
    bm_memcpy_s2d(bm_handle, in[2].device_mem, (void *)attn_mask.data());
    out[1].device_mem = bm_mem_from_device(
        past_key[idx].u.device.device_addr + token_offset, KV_BYTES);
    out[2].device_mem = bm_mem_from_device(
        past_value[idx].u.device.device_addr + token_offset, KV_BYTES);
    // history_k/v: point at the Talker KV cache, NOT the net's pre-allocated
    // input_mems (which are empty/uninitialized).
    in[3].device_mem = past_key[idx];
    in[4].device_mem = past_value[idx];
    net_launch(net, in, out);
    cur_in = net->stages[0].output_mems[0];
  }

  // last hidden -> talker_past_hidden (for next frame's CP)
  bm_memcpy_d2d_byte(bm_handle, talker_past_hidden_mem, 0, cur_in, 0,
                     HIDDEN_SIZE * sizeof(uint16_t));

  // codec_head -> [1,3072] logits -> next code0 (sample/greedy)
  init_tensors(net_lm, in, out);
  in[0].device_mem = talker_past_hidden_mem;
  net_launch(net_lm, in, out);
  int next_code0 = sample_code0(out[0].device_mem);
  visited_tokens[token_length] = next_code0;
  gen_code0_seq.push_back(next_code0);
  token_length++;
  history_length++;
  gen_step++;
  return next_code0;
}

py::array_t<float> Qwen3TTS::forward_mimi_decoder(ArrayInt const &codes) {
  // codes: [16, T] int32. Chunk to MIMI_FRAME (256) frames; concat wav.
  auto p = codes.request();
  assert(p.ndim == 2 && p.shape[0] == 16);
  int T = p.shape[1];
  auto *src = static_cast<int *>(p.ptr);
  int n_chunks = (T + MIMI_FRAME - 1) / MIMI_FRAME;
  int total_samples = T * MIMI_HOP;
  py::array_t<float> wav(total_samples);
  auto w = wav.mutable_unchecked<1>();

  std::vector<bm_tensor_t> in, out;
  init_tensors(net_mimi_decoder, in, out);
  in[0].device_mem = mimi_input_mem;
  out[0].device_mem = mimi_output_mem;

  std::vector<int32_t> chunk(1 * 16 * MIMI_FRAME, 0);
  std::vector<uint16_t> chunk_wav(MIMI_FRAME * MIMI_HOP);
  for (int c = 0; c < n_chunks; c++) {
    int off = c * MIMI_FRAME;
    int n = std::min(MIMI_FRAME, T - off);
    std::fill(chunk.begin(), chunk.end(), 0);
    for (int q = 0; q < 16; q++)
      for (int i = 0; i < n; i++)
        chunk[q * MIMI_FRAME + i] = src[q * T + (off + i)];
    bm_memcpy_s2d(bm_handle, mimi_input_mem, (void *)chunk.data());
    net_launch(net_mimi_decoder, in, out);
    bm_memcpy_d2s(bm_handle, chunk_wav.data(), mimi_output_mem);
    int out_off = off * MIMI_HOP;
    int n_samples = n * MIMI_HOP;
    for (int i = 0; i < n_samples; i++)
      w(out_off + i) = bf16_to_f32(chunk_wav[i]);
  }
  return wav;
}

py::array_t<int> Qwen3TTS::forward_mimi_encoder(ArrayFloat const &wav) {
  // wav: [real_samples] f32 @24kHz. Zero-pad to ENC_T_AUDIO (causal -> exact for
  // the real part), run mimi_encoder, return codes [ENC_N_Q, real_T_12] int32
  // where real_T_12 = real_samples / ENC_DOWNSAMPLE.
  if (!net_mimi_encoder) {
    throw std::runtime_error(
        "forward_mimi_encoder: this bmodel has no mimi_encoder net "
        "(recompile with the ICL converter to enable voice-clone ICL)");
  }
  auto p = wav.request();
  int real_samples = p.size;
  auto *src = static_cast<float *>(p.ptr);
  // zero-pad into a host buffer of ENC_T_AUDIO floats
  std::vector<float> buf(ENC_T_AUDIO, 0.0f);
  if (real_samples > ENC_T_AUDIO) {
    real_samples = ENC_T_AUDIO;  // truncate (shouldn't happen for ~8s refs)
  }
  std::copy(src, src + real_samples, buf.data());

  std::vector<bm_tensor_t> in, out;
  init_tensors(net_mimi_encoder, in, out);
  in[0].device_mem = enc_input_mem;
  out[0].device_mem = enc_output_mem;
  bm_memcpy_s2d(bm_handle, enc_input_mem, (void *)buf.data());
  net_launch(net_mimi_encoder, in, out);
  // Output is f32 (TopK indices cast to f32 by lowering); values are exact
  // integers 0..codebook_size-1. Read as float then round to int32.
  std::vector<float> codes_f(ENC_N_Q * ENC_T_12);
  bm_memcpy_d2s(bm_handle, codes_f.data(), enc_output_mem);
  std::vector<int32_t> codes(ENC_N_Q * ENC_T_12);
  for (size_t i = 0; i < codes_f.size(); i++)
    codes[i] = (int32_t)std::lroundf(codes_f[i]);

  int real_T_12 = real_samples / ENC_DOWNSAMPLE;
  py::array_t<int> result({ENC_N_Q, real_T_12});
  auto r = result.mutable_unchecked<2>();
  for (int q = 0; q < ENC_N_Q; q++)
    for (int i = 0; i < real_T_12; i++)
      r(q, i) = codes[q * ENC_T_12 + i];
  return result;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen3TTS>(m, "Qwen3TTS")
      .def(pybind11::init<>())
      .def("init", &Qwen3TTS::init)
      .def("deinit", &Qwen3TTS::deinit)
      .def("clear_history", &Qwen3TTS::clear_history)
      .def("forward_speaker_encoder", &Qwen3TTS::forward_speaker_encoder)
      .def("text_embedding", &Qwen3TTS::text_embedding)
      .def("codec_embedding", &Qwen3TTS::codec_embedding)
      .def("cp_embed_group", &Qwen3TTS::cp_embed_group)
      .def("set_talker_prefill", &Qwen3TTS::set_talker_prefill)
      .def("set_trailing", &Qwen3TTS::set_trailing)
      .def("forward_first", &Qwen3TTS::forward_first)
      .def("forward_next", &Qwen3TTS::forward_next)
      .def("forward_mimi_decoder", &Qwen3TTS::forward_mimi_decoder)
      .def("forward_mimi_encoder", &Qwen3TTS::forward_mimi_encoder)
      .def("set_sampling", &Qwen3TTS::set_sampling)
      .def("get_frame_codes", &Qwen3TTS::get_frame_codes)
      .def_readonly("SEQLEN", &Qwen3TTS::SEQLEN)
      .def_readonly("MAX_INPUT_LENGTH", &Qwen3TTS::MAX_INPUT_LENGTH)
      .def_readonly("HIDDEN_SIZE", &Qwen3TTS::HIDDEN_SIZE)
      .def_readonly("NUM_LAYERS", &Qwen3TTS::NUM_LAYERS)
      .def_readonly("CP_NUM_CODE_GROUPS", &Qwen3TTS::CP_NUM_CODE_GROUPS)
      .def_readonly("MIMI_FRAME", &Qwen3TTS::MIMI_FRAME)
      .def_readonly("MIMI_HOP", &Qwen3TTS::MIMI_HOP)
      .def_readonly("SPK_MEL_MAX", &Qwen3TTS::SPK_MEL_MAX)
      .def_readonly("ENC_T_AUDIO", &Qwen3TTS::ENC_T_AUDIO)
      .def_readonly("ENC_DOWNSAMPLE", &Qwen3TTS::ENC_DOWNSAMPLE)
      .def_readonly("history_length", &Qwen3TTS::history_length)
      .def_readonly("token_length", &Qwen3TTS::token_length);
}
