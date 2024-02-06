//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include "memory.h"
#include "sentencepiece/sentencepiece_processor.h"
#include "bmruntime_interface.h"
#include <getopt.h>

static const int NUM_LAYERS = 32;
static const int MAX_LEN = 512;
static const float ATTENTION_MASK = -1000.;

static const std::string TOKENIZER_MODEL = "../../bmodel/tokenizer.model";

// #define EXPORT_RESULTS
#ifdef EXPORT_RESULTS
#include "cnpy.h"
static cnpy::npz_t map;

template <typename T>
static void add_array(std::string name, bm_handle_t bm_handle,
                      const bm_device_mem_t &dst) {
  std::vector<T> data(dst.size / sizeof(T));
  bm_memcpy_d2s(bm_handle, data.data(), dst);
  cnpy::npz_add_array(map, name, data);
}

static void save_array(std::string filename) {
  cnpy::npz_save_all(filename, map);
}
#endif

class Llama2 {
public:
  void init(int devid, const std::string model, const std::string tokenizer_path);
  void chat();
  void deinit();
  std::string name;
  std::string history = "";
  int round = 0;
  int token_length;
  int EOS;
  std::string predict_next_token();
  std::string predict_first_token(const std::string &input_str);

private:
  void answer(const std::string &input_str);
  void tokenizer_encode(const std::string &input_str, std::vector<int> &tokens);
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  void step_back(const bm_tensor_t &kv, const bm_tensor_t &kv_cache);
  void load_sentencepiece(const std::string &tokenizer_path);

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  sentencepiece::SentencePieceProcessor sentencepiece;
  const bm_net_info_t *net_blocks[NUM_LAYERS];
  const bm_net_info_t *net_blocks_cache[NUM_LAYERS];
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_lm;
  bm_tensor_t inputs_embed_512, outputs_embed_512;
  bm_tensor_t inputs_lm, outputs_lm;
  bm_tensor_t inputs_pid, next_pid, inputs_attention, next_attention;
  bm_tensor_t past_key[NUM_LAYERS], past_value[NUM_LAYERS];
  bm_tensor_t present_key[NUM_LAYERS], present_value[NUM_LAYERS];
  bm_tensor_t present_key_cache, present_value_cache;
  std::string name_embed;
  std::string name_lm;
  std::string name_blocks[NUM_LAYERS];
  std::string name_blocks_cache[NUM_LAYERS];
};

void Llama2::load_sentencepiece(const std::string &model) {
  printf("Load %s ... ", model.c_str());
  auto status = sentencepiece.Load(model);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(-1);
  }
  EOS = sentencepiece.eos_id();
  printf("Done!\n");
}

void Llama2::init(int devid, const std::string model, const std::string tokenizer_path) {
  load_sentencepiece(tokenizer_path);
  // request bm_handle
  bm_status_t status = bm_dev_request(&bm_handle, devid);
  assert(BM_SUCCESS == status);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model.c_str());
  assert(true == ret);
  printf("Done!\n");
  // net names
  name_embed = "embedding";
  name_lm = "lm_head";
  for (int i = 0; i < NUM_LAYERS; i++) {
    name_blocks[i] = "block_" + std::to_string(i);
    name_blocks_cache[i] = "block_cache_" + std::to_string(i);
  }

  // net infos
  net_embed = bmrt_get_network_info(p_bmrt, name_embed.c_str());
  net_lm = bmrt_get_network_info(p_bmrt, name_lm.c_str());
  for (int i = 0; i < NUM_LAYERS; i++) {
    net_blocks[i] = bmrt_get_network_info(p_bmrt, name_blocks[i].c_str());
    net_blocks_cache[i] =
        bmrt_get_network_info(p_bmrt, name_blocks_cache[i].c_str());
  }

  // net device mem
  ret = bmrt_tensor(&inputs_embed_512, p_bmrt, net_embed->input_dtypes[0],
                    net_embed->stages[1].input_shapes[0]);
  assert(true == ret);

  ret = bmrt_tensor(&outputs_embed_512, p_bmrt, net_embed->output_dtypes[0],
                    net_embed->stages[1].output_shapes[0]);
  assert(true == ret);

  ret = bmrt_tensor(&inputs_pid, p_bmrt, net_blocks[0]->input_dtypes[1],
                    net_blocks[0]->stages[0].input_shapes[1]);
  assert(true == ret);

  ret = bmrt_tensor(&inputs_attention, p_bmrt, net_blocks[0]->input_dtypes[2],
                    net_blocks[0]->stages[0].input_shapes[2]);
  assert(true == ret);

  ret = bmrt_tensor(&next_pid, p_bmrt, net_blocks_cache[0]->input_dtypes[1],
                    net_blocks_cache[0]->stages[0].input_shapes[1]);
  assert(true == ret);

  ret =
      bmrt_tensor(&next_attention, p_bmrt, net_blocks_cache[0]->input_dtypes[2],
                  net_blocks_cache[0]->stages[0].input_shapes[2]);
  assert(true == ret);

  for (int i = 0; i < NUM_LAYERS; i++) {
    ret = bmrt_tensor(&past_key[i], p_bmrt, net_blocks[0]->output_dtypes[1],
                      net_blocks[0]->stages[0].output_shapes[1]);
    assert(true == ret);
    ret = bmrt_tensor(&past_value[i], p_bmrt, net_blocks[0]->output_dtypes[2],
                      net_blocks[0]->stages[0].output_shapes[2]);
    assert(true == ret);
    ret = bmrt_tensor(&present_key[i], p_bmrt, net_blocks[0]->output_dtypes[1],
                      net_blocks[0]->stages[0].output_shapes[1]);
    assert(true == ret);
    ret = bmrt_tensor(&present_value[i], p_bmrt, net_blocks[0]->output_dtypes[2],
                      net_blocks[0]->stages[0].output_shapes[2]);
    assert(true == ret);
  }
  ret = bmrt_tensor(&present_key_cache, p_bmrt, net_blocks_cache[0]->output_dtypes[1],
                    net_blocks_cache[0]->stages[0].output_shapes[1]);
  assert(true == ret);
  ret = bmrt_tensor(&present_value_cache, p_bmrt, net_blocks_cache[0]->output_dtypes[2],
                    net_blocks_cache[0]->stages[0].output_shapes[2]);
  assert(true == ret);

  ret = bmrt_tensor(&inputs_lm, p_bmrt, net_lm->input_dtypes[0],
                    net_lm->stages[0].input_shapes[0]);
  assert(true == ret);
  ret = bmrt_tensor(&outputs_lm, p_bmrt, net_lm->output_dtypes[0],
                    net_lm->stages[0].output_shapes[0]);
  assert(true == ret);
}

void Llama2::deinit() {
  bm_free_device(bm_handle, inputs_embed_512.device_mem);
  bm_free_device(bm_handle, outputs_embed_512.device_mem);
  bm_free_device(bm_handle, inputs_lm.device_mem);
  bm_free_device(bm_handle, outputs_lm.device_mem);
  bm_free_device(bm_handle, inputs_pid.device_mem);
  bm_free_device(bm_handle, next_pid.device_mem);
  bm_free_device(bm_handle, inputs_attention.device_mem);
  bm_free_device(bm_handle, next_attention.device_mem);
  bm_free_device(bm_handle, present_key_cache.device_mem);
  bm_free_device(bm_handle, present_value_cache.device_mem);
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_free_device(bm_handle, past_key[i].device_mem);
    bm_free_device(bm_handle, past_value[i].device_mem);
    bm_free_device(bm_handle, present_key[i].device_mem);
    bm_free_device(bm_handle, present_value[i].device_mem);
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

void Llama2::step_back(const bm_tensor_t &kv, const bm_tensor_t &kv_cache) {
  if (token_length >= MAX_LEN) {
    return;
  }
  auto total_size = bm_mem_get_device_size(kv.device_mem);
  auto bytes = total_size / MAX_LEN;
  auto real_size = (token_length - 1) * bytes;
  auto offset = (MAX_LEN - token_length + 1) * bytes;
  auto mem =
      bm_mem_from_device(bm_mem_get_device_addr(kv.device_mem) + offset, real_size);
  auto mem_cache =
      bm_mem_from_device(bm_mem_get_device_addr(kv_cache.device_mem), bytes);
  auto buffer = new uint8_t[real_size];
  auto buffer_cache = new uint8_t[bytes];
  auto dst = new uint8_t[total_size];
  bm_memcpy_d2s(bm_handle, (void *)buffer, mem);
  bm_memcpy_d2s(bm_handle, (void *)buffer_cache, mem_cache);
  // memset(dst, 0, total_size - real_size);
  memcpy(dst + total_size - real_size - bytes, buffer, real_size);
  memcpy(dst + total_size - bytes, buffer_cache, bytes);
  bm_memcpy_s2d(bm_handle, kv.device_mem, (void *)dst);
  delete[] buffer;
  delete[] buffer_cache;
  delete[] dst;
}

int Llama2::forward_first(std::vector<int> &tokens) {
  int input_ids[MAX_LEN] = {0}; // start token
  int position_id[MAX_LEN] = {0};
  float attention_mask[MAX_LEN * MAX_LEN] = {0};
  token_length = tokens.size();
  
  std::copy(tokens.begin(), tokens.end(), input_ids);
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }

  for (int i = 0; i < MAX_LEN; i++) {
    for (int j = 0; j < MAX_LEN; j++) {
      if (j <= i && i < token_length) {
      } else {
        attention_mask[i * MAX_LEN + j] = ATTENTION_MASK;
      }
    }
  }

  // forward embeding
  bm_memcpy_s2d(bm_handle, inputs_embed_512.device_mem, (void *)input_ids);
  auto ret =
      bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(), &inputs_embed_512, 1,
                            &outputs_embed_512, 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  bm_memcpy_s2d(bm_handle, inputs_pid.device_mem, (void *)position_id);
  bm_memcpy_s2d(bm_handle, inputs_attention.device_mem, (void *)attention_mask);
  auto inputs_embed = outputs_embed_512;
  inputs_embed.shape = net_blocks[0]->stages[0].input_shapes[0];
  bm_tensor_t inputs_block[3] = {inputs_embed, inputs_pid, inputs_attention};
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_tensor_t outputs_block[3] = {inputs_embed, past_key[i], past_value[i]};
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks[i].c_str(), inputs_block, 3,
                                outputs_block, 3, true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
  }
  int bytes = inputs_embed.device_mem.size / MAX_LEN;
  bm_memcpy_d2d_byte(bm_handle, inputs_lm.device_mem, 0,
                     inputs_embed.device_mem, (token_length - 1) * bytes,
                     bytes);
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm, 1,
                              &outputs_lm, 1, true, false);
  bm_thread_sync(bm_handle);
  
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm.device_mem);
  return token;
}

int Llama2::forward_next() {
  float attention_mask[MAX_LEN + 1] = {0};
  for (int i = token_length - 1; i < MAX_LEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;
  // embedding
  outputs_lm.shape = net_embed->stages[0].input_shapes[0];
  auto ret = bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(), &outputs_lm, 1,
                                   &inputs_lm, 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // blocks
  bm_memcpy_s2d(bm_handle, next_attention.device_mem, (void *)attention_mask);
  bm_memcpy_s2d(bm_handle, next_pid.device_mem, (void *)&position_id);
  auto inputs_embed = inputs_lm;
  inputs_embed.shape = net_blocks_cache[0]->stages[0].input_shapes[0];
  int bytes = bm_mem_get_device_size(present_key_cache.device_mem);
  int token_offset = (token_length - 1) * bytes;
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_tensor_t inputs_block[5] = {inputs_embed, next_pid, next_attention,
                                   past_key[i], past_value[i]};
    bm_tensor_t outputs_block[3] = {inputs_embed, present_key_cache, present_value_cache};
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks_cache[i].c_str(),
                                inputs_block, 5, outputs_block, 3, true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
    bm_memcpy_d2d_byte(bm_handle, past_key[i].device_mem, token_offset,
                       present_key_cache.device_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[i].device_mem, token_offset,
                       present_value_cache.device_mem, 0,
                       bytes);
  }
  outputs_lm.shape = net_lm->stages[0].output_shapes[0];
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm, 1,
                              &outputs_lm, 1, true, false);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm.device_mem);
  return token;
}


std::string Llama2::predict_first_token(const std::string &input_str) {
  history = input_str;
  //int tok_num = 1;
  std::vector<int> tokens;
  sentencepiece.Encode(history, &tokens);
  tokens.insert(tokens.begin(), 1);
  if (tokens.empty()) {
    round = 0;
    history = "Sorry: your question is too wierd!!\n";
    return history;
  }
  // make sure token not too large
  if (tokens.size() > MAX_LEN - 10) {
    // reset
    if (round == 0) {
      history = "Error: your question is too large!\n";
      return history;
    }
    round = 0;
    history = "";
    return predict_first_token(input_str);
  }
  int token = forward_first(tokens);
  int pre_token = 0;
  std::string pre_word;
  std::string word;
  std::vector<int> pre_ids = {pre_token};
  std::vector<int> ids = {pre_token,token};
  sentencepiece.Decode(pre_ids, &pre_word);
  sentencepiece.Decode(ids, &word);
  std::string diff = word.substr(pre_word.size());
#ifdef PRINT
  printf("token %d",token);
  printf("diff %s",diff.c_str());
#endif
  history += diff;
  if (token_length < MAX_LEN) {
    token_length++;
  }
  return diff;
}

std::string Llama2::predict_next_token() {
  int pre_token;
  pre_token = 0;
  int token = forward_next();
  if(token == EOS){
    round = 0;
    history = history.substr(history.size()/2);
    return "_GETEOS_";
  }
  std::string pre_word;
  std::string word;
  std::vector<int> pre_ids = {pre_token};
  std::vector<int> ids = {pre_token, token};
  sentencepiece.Decode(pre_ids, &pre_word);
  sentencepiece.Decode(ids, &word);
  std::string diff = word.substr(pre_word.size());
#ifdef PRINT
  printf("token %d",token);
  printf("diff %s",diff.c_str());
#endif
  history += diff;
  if (token_length < MAX_LEN) {
    token_length++;
  }else{
    round = 0;
    return "_GETMAX_";
  }
  return diff;
}


extern "C" {


Llama2 *Llama2_with_devid_and_model(int devid, const char *bmodel_path, const char *tokenizer_path) {
  Llama2 *chat = new Llama2();
  chat->init(devid, bmodel_path, tokenizer_path);
  return chat;
}

void Llama2_delete(Llama2 *chat) { delete chat; }

void Llama2_deinit(Llama2 *chat) { 
  chat->deinit();
}

const char *get_history(Llama2 *chat) {
  std::string str = chat->history;
  return strdup(str.c_str());
}

const char *set_history(Llama2 *chat, const char *history) {
  chat->history = history;
  return strdup(history);
}

const char *Llama2_predict_first_token(Llama2 *chat, const char *input_str) {
  std::string str = chat->predict_first_token(input_str);
  return strdup(str.c_str());
}

const char *Llama2_predict_next_token(Llama2 *chat) {
  std::string str = chat->predict_next_token();
  return strdup(str.c_str());
}

const int get_eos(Llama2 *chat){
  const int res = chat->EOS;
  return res;
}
}
