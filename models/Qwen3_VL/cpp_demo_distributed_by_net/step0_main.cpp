//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// Chip 1: Embedding + VIT + User Interface for distributed Qwen3_VL pipeline.
// Runs embedding/vit locally, sends hidden states to Step1, receives tokens.
//
//===----------------------------------------------------------------------===//

#include "bmruntime_interface.h"
#include "cv_utils.h"
#include "memory.h"
#include "net_helper.hpp"
#include "tokenizers-cpp/tokenizers_cpp.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

using tokenizers::Tokenizer;

static const int VISION_PAD_TOKEN = 151654;
static const int IMAGE_PAD_TOKEN = 151655;
static const int VIDEO_PAD_TOKEN = 151656;

//===------------------------------------------------------------===//
// Helpers
//===------------------------------------------------------------===//
static void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
}

static void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
                      int stage = 0) {
  for (int i = 0; i < net->input_num; i++)
    empty(bm_handle, net->stages[stage].input_mems[i]);
  for (int i = 0; i < net->output_num; i++)
    empty(bm_handle, net->stages[stage].output_mems[i]);
}

static inline std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open [ " << path << " ]" << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

//===------------------------------------------------------------===//
// Step0 Model: Embedding + VIT
//===------------------------------------------------------------===//
class Step0Model {
public:
  void init(int devid, const std::string &model_path);
  void deinit();

  // Embedding for prefill
  void forward_embed(const std::vector<int> &tokens);

  // VIT processing
  void forward_vit(const float *pixel_values,
                   const std::vector<int> &position_ids,
                   const std::vector<int> &pos_idx,
                   const std::vector<float> &pos_weight,
                   const std::vector<int> &grid_thw, int vit_offset);

  // Embedding for decode (single token)
  void forward_embed_cache(int token);

  // Copy hidden states from device to host
  std::vector<uint16_t> get_hidden_states();
  std::vector<uint16_t> get_deepstack(int idx);
  std::vector<uint16_t> get_embed_cache_output();

  int HIDDEN_SIZE;
  int MAX_INPUT_LENGTH;
  int VIT_DIMS;
  int MAX_PATCHES;
  int MAX_PIXELS;
  int token_length = 0;
  bool vit_dynamic;
  bool vit_run = false;
  int num_deepstack;
  bm_handle_t bm_handle;

private:
  void *p_bmrt;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_vit;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> deepstack_buffers;
  bm_device_mem_t embed_cache_out_mem;

  void init_tensors(const bm_net_info_t *net,
                    std::vector<bm_tensor_t> &in_tensors,
                    std::vector<bm_tensor_t> &out_tensors, int stage = 0);
  void net_launch(const bm_net_info_t *net,
                  const std::vector<bm_tensor_t> &in_tensors,
                  std::vector<bm_tensor_t> &out_tensors);
  void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
           int size = 0);
};

void Step0Model::init_tensors(const bm_net_info_t *net,
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

void Step0Model::net_launch(const bm_net_info_t *net,
                            const std::vector<bm_tensor_t> &in_tensors,
                            std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
}

void Step0Model::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                     int size) {
  if (!size)
    size = bm_mem_get_device_size(src);
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Step0Model::init(int devid, const std::string &model_path) {
  printf("[Step0] Device [%d] loading ...\n", devid);
  bm_status_t status = bm_dev_request(&bm_handle, devid);
  assert(BM_SUCCESS == status);

  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  printf("[Step0] Model [%s] loading ...\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  bm_thread_sync(bm_handle);
  printf("[Step0] Model loaded.\n");

  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  assert(net_embed && "embedding network not found");
  assert(net_embed_cache && "embedding_cache network not found");
  assert(net_vit && "vit network not found");

  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_embed->stages[0].output_shapes[0].dims[2];
  MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];
  MAX_PIXELS = MAX_PATCHES * 16 * 16;
  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  vit_dynamic = net_vit->is_dynamic;
  num_deepstack = net_vit->output_num - 1;

  printf("[Step0] HIDDEN_SIZE=%d, MAX_INPUT_LENGTH=%d, VIT_DIMS=%d\n",
         HIDDEN_SIZE, MAX_INPUT_LENGTH, VIT_DIMS);
  printf("[Step0] MAX_PATCHES=%d, num_deepstack=%d\n", MAX_PATCHES,
         num_deepstack);

  // Allocate device buffer
  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);

  for (int i = 0; i < num_deepstack; i++) {
    bm_device_mem_t mem;
    status = bm_malloc_device_byte(bm_handle, &mem, buffer_size);
    assert(BM_SUCCESS == status);
    deepstack_buffers.push_back(mem);
  }

  vit_run = false;
}

void Step0Model::deinit() {
  for (auto &mem : deepstack_buffers)
    bm_free_device(bm_handle, mem);
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void Step0Model::forward_embed(const std::vector<int> &tokens) {
  std::vector<int> padded(MAX_INPUT_LENGTH, 0);
  std::copy(tokens.begin(), tokens.end(), padded.data());
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed, in_tensors, out_tensors);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)padded.data(), MAX_INPUT_LENGTH * sizeof(int));
  net_launch(net_embed, in_tensors, out_tensors);
  empty(bm_handle, dev_buffer);
  d2d(dev_buffer, out_tensors[0].device_mem, 0,
      tokens.size() * HIDDEN_SIZE * sizeof(uint16_t));
  token_length = tokens.size();
  for (auto &mem : deepstack_buffers) {
    empty(bm_handle, mem);
  }
}

void Step0Model::forward_vit(const float *pixel_values,
                             const std::vector<int> &position_ids,
                             const std::vector<int> &pos_idx,
                             const std::vector<float> &pos_weight,
                             const std::vector<int> &grid_thw, int vit_offset) {
  int t = grid_thw[0];
  int h = grid_thw[1];
  int w = grid_thw[2];
  int hw = t * h * w;
  int num_pixels = hw * VIT_DIMS;
  assert((int)position_ids.size() == (hw * 2));
  assert((int)pos_idx.size() == 4 * hw);
  assert((int)pos_weight.size() == 4 * hw);

  empty_net(bm_handle, net_vit);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_vit, in_tensors, out_tensors);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)pixel_values, num_pixels * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                        (void *)position_ids.data(),
                        position_ids.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                        (void *)pos_idx.data(), pos_idx.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[3].device_mem,
                        (void *)pos_weight.data(),
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

  net_launch(net_vit, in_tensors, out_tensors);

  // Merge VIT output into dev_buffer
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = hw / 4 * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                     out_tensors[0].device_mem, 0, vit_size);
  for (int i = 0; i < num_deepstack; i++) {
    bm_memcpy_d2d_byte(bm_handle, deepstack_buffers[i], dst_offset,
                       out_tensors[i + 1].device_mem, 0, vit_size);
  }
  vit_run = true;
}

void Step0Model::forward_embed_cache(int token) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed_cache, in_tensors, out_tensors);
  bm_memcpy_s2d(bm_handle, in_tensors[0].device_mem, (void *)&token);
  net_launch(net_embed_cache, in_tensors, out_tensors);
  embed_cache_out_mem = out_tensors[0].device_mem;
}

std::vector<uint16_t> Step0Model::get_hidden_states() {
  int size = token_length * HIDDEN_SIZE;
  std::vector<uint16_t> data(size);
  bm_memcpy_d2s_partial(bm_handle, (void *)data.data(), dev_buffer,
                        size * sizeof(uint16_t));
  return data;
}

std::vector<uint16_t> Step0Model::get_deepstack(int idx) {
  int size = token_length * HIDDEN_SIZE;
  std::vector<uint16_t> data(size);
  bm_memcpy_d2s_partial(bm_handle, (void *)data.data(), deepstack_buffers[idx],
                        size * sizeof(uint16_t));
  return data;
}

std::vector<uint16_t> Step0Model::get_embed_cache_output() {
  std::vector<uint16_t> data(HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, (void *)data.data(), embed_cache_out_mem,
                        HIDDEN_SIZE * sizeof(uint16_t));
  return data;
}

//===------------------------------------------------------------===//
// Distributed ChatPipe
//===------------------------------------------------------------===//
class ChatPipe {
public:
  Config config;
  ChatPipe(int devid, float video_ratio, float video_fps,
           const std::string &model_path, const std::string &config_path,
           const std::string &step1_host, int step1_port);
  ~ChatPipe();
  void chat();

private:
  Step0Model model;
  int step1_fd = -1;

  // Info received from Step1
  int SEQLEN;
  bool support_history;
  int PREFILL_KV_LENGTH;

  // State
  int history_length = 0;
  int token_length = 0;
  std::vector<int> visited_tokens;

  // Tokenizer
  int ID_IM_END, ID_VISION_START;
  int spatial_merge_size;
  int num_grid_per_side;
  int spatial_merge_unit;
  int tokens_per_second;
  std::unique_ptr<Tokenizer> tok;
  std::unique_ptr<Maker> maker;

  // Network communication
  int send_prefill(const std::vector<int> &position_ids);
  int send_decode(const std::vector<int> &position_ids);
  void send_clear();
  void send_shutdown();

  // Same methods as original pipeline.cpp
  typedef enum { IMAGE, VIDEO, TEXT, UNKNOWN } MediaType;
  MediaType get_media_type(const std::vector<std::string> &medias);
  std::string build_text_prompt(const std::string &input_str);
  std::string build_image_prompt(const std::string &input_str,
                                 const std::vector<std::vector<int>> &grid_thw);
  std::string build_video_prompt(const std::string &input_str,
                                 const std::vector<int> &grid_thw,
                                 const std::vector<double> &timestamps);
  std::vector<std::vector<int>>
  rot_pos(const std::vector<std::vector<int>> &grid_thw);
  std::vector<std::vector<int>>
  get_rope_index(const std::vector<int> &input_ids,
                 const std::vector<std::vector<int>> &grid_thw, int pad_id);
  void fast_pos_embed_interpolate(const std::vector<int> &grid_thw,
                                  std::vector<int> &idx_out,
                                  std::vector<float> &weight_out);
  std::vector<int> find_token_offset(const std::vector<int> &input_ids,
                                     int pad_id);
  std::vector<int> get_position_ids(int token_len);
  void vit_process_image(std::vector<float> &pixel_values, int vit_offset);
  void vit_process_video(std::vector<float> &pixel_values,
                         std::vector<int> &vit_offset);
  std::vector<int> encode_input(const std::string &sentence_input);
  void print_chat_instructions();
  int forward_prefill(std::vector<int> &position_ids_1d, int &max_posid,
                      int &history_max_posid);
};

// ===== Helper functions (same as pipeline.cpp) =====

static inline std::vector<float> linspace_inclusive(float start, float end,
                                                    int num) {
  std::vector<float> out;
  out.reserve(num);
  if (num == 1) {
    out.push_back(start);
    return out;
  }
  float step = (end - start) / float(num - 1);
  for (int i = 0; i < num; ++i)
    out.push_back(start + step * i);
  return out;
}

static std::vector<size_t> argwhere(const std::vector<int> &vec, int value) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] == value)
      indices.push_back(i);
  }
  return indices;
}

static std::vector<int> arange(int start, int end) {
  std::vector<int> result;
  for (int i = start; i < end; ++i)
    result.push_back(i);
  return result;
}

static int vec_max(const std::vector<int> &vec) {
  int max_val = vec[0];
  for (int val : vec) {
    if (val > max_val)
      max_val = val;
  }
  return max_val;
}

static std::vector<std::vector<int>>
cat(const std::vector<std::vector<std::vector<int>>> &vecs, int dim) {
  if (dim == 1) {
    std::vector<std::vector<int>> result(3);
    for (const auto &vec : vecs) {
      for (int i = 0; i < 3; ++i)
        result[i].insert(result[i].end(), vec[i].begin(), vec[i].end());
    }
    return result;
  }
  return {};
}

static std::string strip(const std::string &s) {
  const std::string WHITESPACE = " \n\r\t\f\v";
  size_t start = s.find_first_not_of(WHITESPACE);
  if (start == std::string::npos)
    return "";
  size_t end = s.find_last_not_of(WHITESPACE);
  return s.substr(start, end - start + 1);
}

static std::vector<std::string> splitString(const std::string &s) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    result.push_back(strip(item));
  return result;
}

static std::string format_seconds(double curr_time) {
  std::ostringstream oss;
  oss << "<" << std::fixed << std::setprecision(1) << curr_time << " seconds>";
  return oss.str();
}

static std::vector<double> calculate_timestamps(const std::vector<int> &indices,
                                                double video_fps,
                                                int merge_size = 2) {
  std::vector<double> timestamps(indices.size());
  for (size_t i = 0; i < indices.size(); ++i)
    timestamps[i] = static_cast<double>(indices[i]) / video_fps;
  std::vector<double> merged;
  for (size_t i = 0; i < timestamps.size(); i += merge_size) {
    size_t j = i + merge_size - 1;
    merged.push_back((timestamps[i] + timestamps[j]) / 2.0);
  }
  return merged;
}

// ===== ChatPipe implementation =====

ChatPipe::ChatPipe(int devid, float video_ratio, float video_fps,
                   const std::string &model_path,
                   const std::string &config_path,
                   const std::string &step1_host, int step1_port) {
  model.init(devid, model_path);
  spatial_merge_size = 2;
  spatial_merge_unit = spatial_merge_size * spatial_merge_size;
  tokens_per_second = 2;
  num_grid_per_side = 48;

  // Load tokenizer
  std::cout << "Processor [" << config_path << "] loading .... ";
  auto blob = LoadBytesFromFile(config_path + "/tokenizer.json");
  tok = Tokenizer::FromBlobJSON(blob);
  ID_IM_END = tok->TokenToId("<|im_end|>");
  ID_VISION_START = tok->TokenToId("<|vision_start|>");
  std::cout << "Done!" << std::endl;

  config.temporal_patch_size = 2;
  config.spatial_merge_size = 2;
  config.patch_size = 16;
  config.MAX_INPUT_LENGTH = model.MAX_INPUT_LENGTH;
  config.video_ratio = video_ratio;
  config.MAX_PIXELS = model.MAX_PIXELS;
  config.MAX_PATCHES = model.MAX_PATCHES;
  config.MIN_PIXELS = 64 * 32 * 32;
  config.video_fps = video_fps;
  maker = std::make_unique<Maker>(config);

  // Connect to Step1
  printf("[Step0] Connecting to Step1 at %s:%d ...\n", step1_host.c_str(),
         step1_port);
  step1_fd = connect_to(step1_host, step1_port);

  // Receive handshake from Step1
  struct {
    int32_t hidden_size;
    int32_t seqlen;
    int32_t max_input_length;
    int32_t support_history;
    int32_t prefill_kv_length;
  } handshake;
  recv_all(step1_fd, &handshake, sizeof(handshake));
  assert(handshake.hidden_size == model.HIDDEN_SIZE);
  assert(handshake.max_input_length == model.MAX_INPUT_LENGTH);
  SEQLEN = handshake.seqlen;
  support_history = handshake.support_history != 0;
  PREFILL_KV_LENGTH = handshake.prefill_kv_length;
  config.SEQLEN = SEQLEN;

  printf("[Step0] Step1 info: SEQLEN=%d, support_history=%d, "
         "PREFILL_KV_LENGTH=%d\n",
         SEQLEN, (int)support_history, PREFILL_KV_LENGTH);

  visited_tokens.resize(SEQLEN, 0);
  history_length = 0;
}

ChatPipe::~ChatPipe() {
  if (step1_fd >= 0) {
    send_shutdown();
    close(step1_fd);
  }
}

int ChatPipe::send_prefill(const std::vector<int> &position_ids) {
  // Get hidden states from device
  bm_thread_sync(model.bm_handle);
  auto hidden_states = model.get_hidden_states();

  // Get deepstack buffers if VIT was run
  int num_ds = model.vit_run ? model.num_deepstack : 0;
  std::vector<std::vector<uint16_t>> deepstacks;
  for (int i = 0; i < num_ds; i++) {
    deepstacks.push_back(model.get_deepstack(i));
  }
  model.vit_run = false;

  // Build and send PrefillMeta
  PrefillMeta meta;
  meta.msg_type = MSG_PREFILL;
  meta.token_length = model.token_length;
  meta.num_deepstack = num_ds;
  meta.position_ids_count = position_ids.size();
  meta.hidden_size = model.HIDDEN_SIZE;
  meta.visited_token_count =
      model.token_length; // send input tokens for sampling

  send_all(step1_fd, &meta, sizeof(meta));

  // Send position_ids
  send_all(step1_fd, position_ids.data(), position_ids.size() * sizeof(int));

  // Send visited_tokens (the input tokens for sampling)
  send_all(step1_fd, visited_tokens.data(),
           meta.visited_token_count * sizeof(int));

  // Send hidden_states
  send_all(step1_fd, hidden_states.data(),
           hidden_states.size() * sizeof(uint16_t));

  // Send deepstack buffers
  for (int i = 0; i < num_ds; i++) {
    send_all(step1_fd, deepstacks[i].data(),
             deepstacks[i].size() * sizeof(uint16_t));
  }

  // Receive token + history_length
  TokenWithHistory tw;
  recv_all(step1_fd, &tw, sizeof(tw));

  // Update local state
  visited_tokens[token_length] = tw.token;
  token_length++;
  history_length = tw.history_length;

  return tw.token;
}

int ChatPipe::send_decode(const std::vector<int> &position_ids) {
  assert(position_ids.size() == 3);

  // Run embed_cache on Step0
  int last_token = visited_tokens[token_length - 1];
  model.forward_embed_cache(last_token);
  bm_thread_sync(model.bm_handle);
  auto hidden_state = model.get_embed_cache_output();

  // Build and send DecodeMeta
  DecodeMeta meta;
  meta.msg_type = MSG_DECODE;
  meta.position_ids[0] = position_ids[0];
  meta.position_ids[1] = position_ids[1];
  meta.position_ids[2] = position_ids[2];
  meta.hidden_size = model.HIDDEN_SIZE;

  send_all(step1_fd, &meta, sizeof(meta));
  send_all(step1_fd, hidden_state.data(),
           hidden_state.size() * sizeof(uint16_t));

  // Receive token + history_length
  TokenWithHistory tw;
  recv_all(step1_fd, &tw, sizeof(tw));
  history_length = tw.history_length;

  // Update local state
  visited_tokens[token_length] = tw.token;
  token_length++;

  return tw.token;
}

void ChatPipe::send_clear() {
  SimpleMsg msg;
  msg.msg_type = MSG_CLEAR;
  send_all(step1_fd, &msg, sizeof(msg));
  // Wait for ack
  TokenWithHistory tw;
  recv_all(step1_fd, &tw, sizeof(tw));
  history_length = 0;
  token_length = 0;
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
}

void ChatPipe::send_shutdown() {
  SimpleMsg msg;
  msg.msg_type = MSG_SHUTDOWN;
  send_all(step1_fd, &msg, sizeof(msg));
}

int ChatPipe::forward_prefill(std::vector<int> &position_ids_1d, int &max_posid,
                              int &history_max_posid) {
  if (history_length == 0 || !support_history) {
    history_max_posid = 0;
    return send_prefill(position_ids_1d);
  }

  if (history_length + model.token_length + 128 > SEQLEN ||
      history_length > PREFILL_KV_LENGTH) {
    std::cerr << "Warning: History is full, clearing." << std::endl;
    send_clear();
    history_max_posid = 0;
    return send_prefill(position_ids_1d);
  }

  for (auto &x : position_ids_1d)
    x += history_max_posid;
  max_posid += history_max_posid;
  return send_prefill(position_ids_1d);
}

// ===== ChatPipe methods from pipeline.cpp (unchanged logic) =====

ChatPipe::MediaType
ChatPipe::get_media_type(const std::vector<std::string> &medias) {
  if (medias.empty() || medias[0].empty())
    return TEXT;
  auto type = UNKNOWN;
  for (auto &m : medias) {
    std::string ext = m.substr(m.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" ||
        ext == "webp") {
      if (type == UNKNOWN)
        type = IMAGE;
      else if (type != IMAGE)
        return UNKNOWN;
    } else if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" ||
               ext == "flv" || ext == "wmv") {
      if (type == UNKNOWN)
        type = VIDEO;
      else if (type != VIDEO)
        return UNKNOWN;
    }
  }
  return type;
}

std::vector<int> ChatPipe::get_position_ids(int token_len) {
  std::vector<int> position_ids(token_len * 3);
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < token_len; ++i)
      position_ids[j * token_len + i] = i;
  return position_ids;
}

std::string ChatPipe::build_text_prompt(const std::string &input_str) {
  return "<|im_start|>user\n" + input_str +
         "\n<|im_end|>\n<|im_start|>assistant\n";
}

std::string
ChatPipe::build_image_prompt(const std::string &input_str,
                             const std::vector<std::vector<int>> &grid_thw) {
  std::string prompt = "<|im_start|>user\n";
  int num_images = grid_thw.size();
  for (int i = 0; i < num_images; i++) {
    int h = grid_thw[i][1];
    int w = grid_thw[i][2];
    int pad_len = h * w / 4;
    prompt += "<|vision_start|>";
    for (int j = 0; j < pad_len; j++)
      prompt += "<|image_pad|>";
    prompt += "<|vision_end|>";
  }
  prompt += input_str + "<|im_end|>\n<|im_start|>assistant\n";
  return prompt;
}

std::string
ChatPipe::build_video_prompt(const std::string &input_str,
                             const std::vector<int> &thw,
                             const std::vector<double> &timestamps) {
  std::string prompt = "<|im_start|>user\n";
  int t = thw[0], h = thw[1], w = thw[2];
  int pad_len = h * w / 4;
  for (int i = 0; i < t; i++) {
    prompt += format_seconds(timestamps[i]);
    prompt += "<|vision_start|>";
    for (int j = 0; j < pad_len; j++)
      prompt += "<|video_pad|>";
    prompt += "<|vision_end|>";
  }
  prompt += input_str + "<|im_end|>\n<|im_start|>assistant\n";
  return prompt;
}

void ChatPipe::fast_pos_embed_interpolate(const std::vector<int> &grid_thw,
                                          std::vector<int> &idx_out,
                                          std::vector<float> &weight_out) {
  if (grid_thw.empty())
    throw std::invalid_argument("grid_thw must contain at least one element");
  int t = 1, h = grid_thw[1], w = grid_thw[2];
  if (h <= 0 || w <= 0 || t <= 0)
    throw std::invalid_argument("t, h, w must be positive");

  auto h_idxs = linspace_inclusive(0.0f, float(num_grid_per_side - 1), h);
  auto w_idxs = linspace_inclusive(0.0f, float(num_grid_per_side - 1), w);

  std::vector<int> h_floor(h), h_ceil(h), w_floor(w), w_ceil(w);
  std::vector<float> dh(h), dw(w);
  for (int i = 0; i < h; ++i) {
    int f = static_cast<int>(h_idxs[i]);
    h_floor[i] = f;
    h_ceil[i] = std::min(f + 1, num_grid_per_side - 1);
    dh[i] = h_idxs[i] - float(f);
  }
  for (int j = 0; j < w; ++j) {
    int f = static_cast<int>(w_idxs[j]);
    w_floor[j] = f;
    w_ceil[j] = std::min(f + 1, num_grid_per_side - 1);
    dw[j] = w_idxs[j] - float(f);
  }

  std::vector<int> base_h(h), base_h_ceil(h);
  for (int i = 0; i < h; ++i) {
    base_h[i] = h_floor[i] * num_grid_per_side;
    base_h_ceil[i] = h_ceil[i] * num_grid_per_side;
  }

  std::vector<int> idx00, idx01, idx10, idx11;
  std::vector<float> w00, w01, w10, w11;
  idx00.reserve(h * w);
  idx01.reserve(h * w);
  idx10.reserve(h * w);
  idx11.reserve(h * w);
  w00.reserve(h * w);
  w01.reserve(h * w);
  w10.reserve(h * w);
  w11.reserve(h * w);

  for (int i = 0; i < h; ++i) {
    float dh_i = dh[i], one_dh_i = 1.0f - dh_i;
    int bi = base_h[i], bic = base_h_ceil[i];
    for (int j = 0; j < w; ++j) {
      float dw_j = dw[j], one_dw_j = 1.0f - dw_j;
      idx00.push_back(bi + w_floor[j]);
      idx01.push_back(bi + w_ceil[j]);
      idx10.push_back(bic + w_floor[j]);
      idx11.push_back(bic + w_ceil[j]);
      w00.push_back(one_dh_i * one_dw_j);
      w01.push_back(one_dh_i * dw_j);
      w10.push_back(dh_i * one_dw_j);
      w11.push_back(dh_i * dw_j);
    }
  }

  int msize = spatial_merge_size;
  int H_blk = h / msize, W_blk = w / msize;
  idx_out.resize(t * h * w * 4);
  weight_out.resize(t * h * w * 4);
  std::vector<int> out_order;
  out_order.reserve(h * w);
  for (int i_blk = 0; i_blk < H_blk; ++i_blk)
    for (int j_blk = 0; j_blk < W_blk; ++j_blk)
      for (int i2 = 0; i2 < msize; ++i2)
        for (int j2 = 0; j2 < msize; ++j2)
          out_order.push_back((i_blk * msize + i2) * w + (j_blk * msize + j2));

  for (int k = 0; k < (int)out_order.size(); ++k) {
    int src = out_order[k], base = k * 4;
    idx_out[base] = idx00[src];
    idx_out[base + 1] = idx01[src];
    idx_out[base + 2] = idx10[src];
    idx_out[base + 3] = idx11[src];
    weight_out[base] = w00[src];
    weight_out[base + 1] = w01[src];
    weight_out[base + 2] = w10[src];
    weight_out[base + 3] = w11[src];
  }
}

std::vector<std::vector<int>>
ChatPipe::rot_pos(const std::vector<std::vector<int>> &grid_thw) {
  std::vector<std::vector<int>> pos_ids;
  for (const auto &thw : grid_thw) {
    int t = thw[0], h = thw[1], w = thw[2];
    std::vector<int> hpos_ids;
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < w; ++j)
        hpos_ids.push_back(i);

    int h_merged = h / spatial_merge_size, w_merged = w / spatial_merge_size;
    std::vector<int> reshaped_hpos;
    for (int i = 0; i < h_merged; ++i)
      for (int j = 0; j < w_merged; ++j)
        for (int k = 0; k < spatial_merge_size; ++k)
          for (int l = 0; l < spatial_merge_size; ++l)
            reshaped_hpos.push_back(hpos_ids[(i * spatial_merge_size + k) * w +
                                             (j * spatial_merge_size + l)]);

    std::vector<int> wpos_ids;
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < w; ++j)
        wpos_ids.push_back(j);

    std::vector<int> reshaped_wpos;
    for (int i = 0; i < h_merged; ++i)
      for (int j = 0; j < w_merged; ++j)
        for (int k = 0; k < spatial_merge_size; ++k)
          for (int l = 0; l < spatial_merge_size; ++l)
            reshaped_wpos.push_back(wpos_ids[(i * spatial_merge_size + k) * w +
                                             (j * spatial_merge_size + l)]);

    std::vector<std::vector<int>> merged;
    for (size_t i = 0; i < reshaped_hpos.size(); ++i)
      merged.push_back({reshaped_hpos[i], reshaped_wpos[i]});

    for (int i = 0; i < t; ++i)
      pos_ids.insert(pos_ids.end(), merged.begin(), merged.end());
  }
  return pos_ids;
}

std::vector<std::vector<int>>
ChatPipe::get_rope_index(const std::vector<int> &input_ids,
                         const std::vector<std::vector<int>> &grid_thw,
                         int pad_id) {
  size_t seq_length = input_ids.size();
  std::vector<std::vector<int>> position_ids(3,
                                             std::vector<int>(seq_length, 1));
  std::vector<size_t> vision_start_indices =
      argwhere(input_ids, ID_VISION_START);
  int image_nums = vision_start_indices.size();

  std::vector<std::vector<std::vector<int>>> llm_pos_ids_list;
  size_t st = 0;
  int remain_images = image_nums;
  int second_per_grid_t = pad_id == VIDEO_PAD_TOKEN ? 1 : 0;

  for (int img_idx = 0; img_idx < image_nums; ++img_idx) {
    size_t ed_image = input_ids.size();
    if (remain_images > 0) {
      auto it = std::find(input_ids.begin() + st, input_ids.end(), pad_id);
      if (it != input_ids.end())
        ed_image = it - input_ids.begin();
    }
    int t, h, w;
    if (pad_id == IMAGE_PAD_TOKEN) {
      t = grid_thw[img_idx][0];
      h = grid_thw[img_idx][1];
      w = grid_thw[img_idx][2];
    } else {
      t = 1;
      h = grid_thw[0][1];
      w = grid_thw[0][2];
    }
    --remain_images;
    size_t ed = ed_image;
    int llm_grid_t = t;
    int llm_grid_h = h / spatial_merge_size;
    int llm_grid_w = w / spatial_merge_size;
    size_t text_len = ed - st;

    int st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      int max_val = 0;
      for (const auto &row : llm_pos_ids_list.back()) {
        int row_max = vec_max(row);
        if (row_max > max_val)
          max_val = row_max;
      }
      st_idx = max_val + 1;
    }

    std::vector<std::vector<int>> text_pos(3);
    std::vector<int> text_range = arange(0, text_len);
    for (int j = 0; j < 3; ++j) {
      std::vector<int> temp(text_range);
      for (int &val : temp)
        val += st_idx;
      text_pos[j] = temp;
    }
    llm_pos_ids_list.push_back(text_pos);

    std::vector<int> t_index;
    for (int i = 0; i < llm_grid_t; i++)
      t_index.insert(t_index.end(), llm_grid_h * llm_grid_w,
                     i * second_per_grid_t * tokens_per_second);
    std::vector<int> h_index;
    for (int n = 0; n < llm_grid_t; ++n)
      for (int p = 0; p < llm_grid_h; ++p)
        for (int q = 0; q < llm_grid_w; ++q)
          h_index.push_back(p);
    std::vector<int> w_index;
    for (int n = 0; n < llm_grid_t; ++n)
      for (int p = 0; p < llm_grid_h; ++p)
        for (int q = 0; q < llm_grid_w; ++q)
          w_index.push_back(q);

    std::vector<std::vector<int>> grid_pos = {t_index, h_index, w_index};
    for (auto &row : grid_pos)
      for (int &val : row)
        val += text_len + st_idx;
    llm_pos_ids_list.push_back(grid_pos);
    st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
  }

  if (st < input_ids.size()) {
    int st_idx = 0;
    if (!llm_pos_ids_list.empty()) {
      int max_val = 0;
      for (const auto &row : llm_pos_ids_list.back()) {
        int row_max = vec_max(row);
        if (row_max > max_val)
          max_val = row_max;
      }
      st_idx = max_val + 1;
    }
    size_t text_len = input_ids.size() - st;
    std::vector<std::vector<int>> text_pos(3);
    std::vector<int> text_range = arange(0, text_len);
    for (int j = 0; j < 3; ++j) {
      std::vector<int> temp(text_range);
      for (int &val : temp)
        val += st_idx;
      text_pos[j] = temp;
    }
    llm_pos_ids_list.push_back(text_pos);
  }

  return cat(llm_pos_ids_list, 1);
}

std::vector<int> ChatPipe::find_token_offset(const std::vector<int> &input_ids,
                                             int pad_id) {
  std::vector<int> offsets;
  for (int i = 0; i < (int)input_ids.size(); ++i)
    if (input_ids[i] == pad_id)
      offsets.push_back(i);
  return offsets;
}

void ChatPipe::vit_process_image(std::vector<float> &pixel_values,
                                 int vit_offset) {
  std::vector<std::vector<int>> grid_thw = {config.grid_thw};
  auto pos_ids_vec = rot_pos(grid_thw);
  std::vector<int> position_ids;
  for (const auto &v : pos_ids_vec)
    position_ids.insert(position_ids.end(), v.begin(), v.end());
  std::vector<int> pos_ids;
  std::vector<float> pos_weight;
  fast_pos_embed_interpolate(config.grid_thw, pos_ids, pos_weight);
  model.forward_vit(pixel_values.data(), position_ids, pos_ids, pos_weight,
                    config.grid_thw, vit_offset);
}

void ChatPipe::vit_process_video(std::vector<float> &pixel_values,
                                 std::vector<int> &vit_offset) {
  int t = config.grid_thw[0], h = config.grid_thw[1], w = config.grid_thw[2];
  assert(t == (int)vit_offset.size());
  std::vector<int> pos_ids;
  std::vector<float> pos_weight;
  fast_pos_embed_interpolate(config.grid_thw, pos_ids, pos_weight);
  std::vector<std::vector<int>> grid_thw = {{1, h, w}};
  auto pos_ids_vec = rot_pos(grid_thw);
  std::vector<int> position_ids;
  for (const auto &v : pos_ids_vec)
    position_ids.insert(position_ids.end(), v.begin(), v.end());
  for (int i = 0; i < t; i++)
    model.forward_vit(pixel_values.data() + i * h * w * model.VIT_DIMS,
                      position_ids, pos_ids, pos_weight, grid_thw[0],
                      vit_offset[i] + 1);
}

std::vector<int> ChatPipe::encode_input(const std::string &sentence_input) {
  return tok->Encode(sentence_input);
}

void ChatPipe::print_chat_instructions() {
  std::cout
      << "\n================================================================="
         "\n"
      << "1. If you want to quit, please enter one of [q, quit, exit]\n"
      << "2. To create a new chat session, please enter one of [clear, new]\n"
      << "=================================================================\n"
      << "[Distributed Mode: Step0=embed+vit, Step1=block, Step2=lmhead]\n";
}

// ===== Main chat loop =====

void ChatPipe::chat() {
  using clock = std::chrono::steady_clock;
  print_chat_instructions();
  int history_max_posid = 0;

  while (true) {
    std::string input_str;
    int token = 0;
    int max_posid = 0;
    std::cout << "\nQuestion: ";
    std::getline(std::cin, input_str);
    input_str = strip(input_str);
    if (input_str == "exit" || input_str == "q" || input_str == "quit")
      break;
    if (input_str == "clear" || input_str == "c" || input_str == "new") {
      send_clear();
      history_max_posid = 0;
      std::cout << "Chat history cleared." << std::endl;
      continue;
    }

    std::string media_path;
    std::cout << "\nImage or Video Path: ";
    std::getline(std::cin, media_path);
    auto medias = splitString(media_path);
    auto media_type = get_media_type(medias);
    if (media_type == UNKNOWN) {
      std::cout << "Unsupported media type." << std::endl;
      continue;
    }
    if (media_type != TEXT) {
      for (auto &m : medias) {
        if (!std::filesystem::exists(m)) {
          std::cerr << "File does not exist: " << m << std::endl;
          continue;
        }
      }
    }

    std::cout << "\nAnswer:\n";
    int64_t duration_prefill = 0, duration_vit = 0, duration_decode = 0;
    int input_token_num = 0;
    clock::time_point clock_start;

    switch (media_type) {
    case IMAGE: {
      int num_medias = medias.size();
      std::vector<std::vector<float>> pixel_values_arr(num_medias);
      std::vector<std::vector<int>> grid_thws;
      for (int i = 0; i < num_medias; ++i) {
        auto ret = process_image(pixel_values_arr[i], medias[i], config);
        if (!ret) {
          std::cerr << "Error processing image: " << medias[i] << std::endl;
          continue;
        }
        grid_thws.push_back(config.grid_thw);
      }
      std::string sentence_input = build_image_prompt(input_str, grid_thws);
      std::vector<int> tokens = encode_input(sentence_input);
      if ((int)tokens.size() > model.MAX_INPUT_LENGTH) {
        std::cerr << "Input tokens exceed max: " << model.MAX_INPUT_LENGTH
                  << std::endl;
        continue;
      }
      input_token_num = tokens.size();
      auto vit_offset = find_token_offset(tokens, ID_VISION_START);

      // Store tokens as visited_tokens
      token_length = tokens.size();
      std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
      std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

      clock_start = clock::now();
      model.forward_embed(tokens);
      auto clock_vit_start = clock::now();
      for (int i = 0; i < num_medias; ++i)
        vit_process_image(pixel_values_arr[i], vit_offset[i] + 1);
      auto clock_vit_end = clock::now();
      duration_vit = std::chrono::duration_cast<std::chrono::milliseconds>(
                         clock_vit_end - clock_vit_start)
                         .count();
      auto position_ids = get_rope_index(tokens, grid_thws, IMAGE_PAD_TOKEN);
      for (int val : position_ids[0])
        if (val > max_posid)
          max_posid = val;
      std::vector<int> position_ids_1d;
      for (const auto &dim : position_ids)
        position_ids_1d.insert(position_ids_1d.end(), dim.begin(), dim.end());
      token = forward_prefill(position_ids_1d, max_posid, history_max_posid);
    } break;
    case VIDEO: {
      std::vector<float> pixel_values;
      std::vector<int> frame_indices;
      double fps;
      auto ret =
          process_video(pixel_values, frame_indices, medias[0], config, fps);
      if (!ret) {
        std::cerr << "Error processing video: " << medias[0] << std::endl;
        continue;
      }
      auto timestamps =
          calculate_timestamps(frame_indices, fps, config.spatial_merge_size);
      std::string sentence_input =
          build_video_prompt(input_str, config.grid_thw, timestamps);
      std::vector<int> tokens = encode_input(sentence_input);
      if ((int)tokens.size() > model.MAX_INPUT_LENGTH) {
        std::cerr << "Input tokens exceed max: " << model.MAX_INPUT_LENGTH
                  << std::endl;
        continue;
      }
      input_token_num = tokens.size();
      auto vit_offset = find_token_offset(tokens, ID_VISION_START);

      token_length = tokens.size();
      std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
      std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

      clock_start = clock::now();
      model.forward_embed(tokens);
      auto clock_vit_start = clock::now();
      vit_process_video(pixel_values, vit_offset);
      auto clock_vit_end = clock::now();
      duration_vit = std::chrono::duration_cast<std::chrono::milliseconds>(
                         clock_vit_end - clock_vit_start)
                         .count();
      auto position_ids =
          get_rope_index(tokens, {config.grid_thw}, VIDEO_PAD_TOKEN);
      for (int val : position_ids[0])
        if (val > max_posid)
          max_posid = val;
      std::vector<int> position_ids_1d;
      for (const auto &dim : position_ids)
        position_ids_1d.insert(position_ids_1d.end(), dim.begin(), dim.end());
      token = forward_prefill(position_ids_1d, max_posid, history_max_posid);
    } break;
    case TEXT: {
      std::string sentence_input = build_text_prompt(input_str);
      std::vector<int> tokens = encode_input(sentence_input);
      if ((int)tokens.size() > model.MAX_INPUT_LENGTH) {
        std::cerr << "Input tokens exceed max: " << model.MAX_INPUT_LENGTH
                  << std::endl;
        continue;
      }
      input_token_num = tokens.size();

      token_length = tokens.size();
      std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
      std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

      clock_start = clock::now();
      model.forward_embed(tokens);
      auto position_ids_1d = get_position_ids(tokens.size());
      max_posid = tokens.size() - 1;
      token = forward_prefill(position_ids_1d, max_posid, history_max_posid);
    } break;
    default:
      std::cerr << "Unsupported media type." << std::endl;
      continue;
    }

    auto clock_prefill = clock::now();
    duration_prefill = std::chrono::duration_cast<std::chrono::milliseconds>(
                           clock_prefill - clock_start)
                           .count();

    // Decode loop
    std::vector<int> full_word_tokens;
    std::string text;
    int output_token_num = 0;
    while (token != ID_IM_END && history_length < SEQLEN) {
      full_word_tokens.push_back(token);
      std::string word = tok->Decode(full_word_tokens);
      if (word.find("�") == std::string::npos) {
        if (full_word_tokens.size() == 1) {
          std::string pre_word = word;
          std::vector<int> double_token = {token, token};
          word = tok->Decode(double_token).substr(pre_word.length());
        }
        text += word;
        std::cout << word << std::flush;
        full_word_tokens.clear();
      }
      max_posid++;
      std::vector<int> following_position_ids = {max_posid, max_posid,
                                                 max_posid};
      token = send_decode(following_position_ids);
      output_token_num++;
    }
    history_max_posid = max_posid + 2;
    std::cout << std::endl;
    auto clock_end = clock::now();
    duration_decode = std::chrono::duration_cast<std::chrono::milliseconds>(
                          clock_end - clock_prefill)
                          .count();
    std::cout << "FTL: " << duration_prefill / 1000.0f << " s" << std::endl;
    if (output_token_num > 0) {
      std::cout << "TPS: " << output_token_num * 1000.0f / duration_decode
                << " tokens/s" << std::endl;
    }
    if (duration_vit > 0) {
      std::cout << "Vision [" << config.grid_thw[0] << ", "
                << config.grid_thw[1] << ", " << config.grid_thw[2]
                << "]: " << duration_vit / 1000.0f << " s" << std::endl;
    }
    std::cout << "Input Tokens: " << input_token_num
              << ", Output Tokens: " << output_token_num + 1 << std::endl;
  }
}

//===------------------------------------------------------------===//
// Main
//===------------------------------------------------------------===//
void Usage() {
  printf("Step0 (Embed+VIT Master) Usage:\n"
         "  -m, --model       : Embed+VIT bmodel path\n"
         "  -c, --config      : Config path (tokenizer, etc.)\n"
         "  -d, --devid       : Device ID (default 0)\n"
         "  -r, --video_ratio : Video ratio (default 0.25)\n"
         "  -f, --video_fps   : Video fps (default 1.0)\n"
         "  --step1_host      : Step1 host (default 127.0.0.1)\n"
         "  --step1_port      : Step1 port (default 10001)\n"
         "  -h, --help        : Show help\n");
}

int main(int argc, char *argv[]) {
  std::string model_path;
  std::string config_path;
  int devid = 0;
  float video_ratio = 0.25f;
  float video_fps = 1.0f;
  std::string step1_host = "127.0.0.1";
  int step1_port = 10001;

  struct option long_opts[] = {{"model", required_argument, nullptr, 'm'},
                               {"config", required_argument, nullptr, 'c'},
                               {"devid", required_argument, nullptr, 'd'},
                               {"video_ratio", required_argument, nullptr, 'r'},
                               {"video_fps", required_argument, nullptr, 'f'},
                               {"step1_host", required_argument, nullptr, 1},
                               {"step1_port", required_argument, nullptr, 2},
                               {"help", no_argument, nullptr, 'h'},
                               {nullptr, 0, nullptr, 0}};
  int opt;
  while ((opt = getopt_long(argc, argv, "m:c:d:r:f:h", long_opts, nullptr)) !=
         -1) {
    switch (opt) {
    case 'm':
      model_path = optarg;
      break;
    case 'c':
      config_path = optarg;
      break;
    case 'd':
      devid = atoi(optarg);
      break;
    case 'r':
      video_ratio = atof(optarg);
      break;
    case 'f':
      video_fps = atof(optarg);
      break;
    case 1:
      step1_host = optarg;
      break;
    case 2:
      step1_port = atoi(optarg);
      break;
    case 'h':
      Usage();
      return 0;
    default:
      Usage();
      return 1;
    }
  }
  if (model_path.empty() || config_path.empty()) {
    Usage();
    return 1;
  }
  assert(video_fps > 0);

  ChatPipe pipeline(devid, video_ratio, video_fps, model_path, config_path,
                    step1_host, step1_port);
  pipeline.chat();
  return 0;
}
