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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "memory.h"
#include "sentencepiece/sentencepiece_processor.h"
#include "bmruntime_interface.h"
#include <getopt.h>
#include <stdio.h>
#include <inttypes.h>

static const uint16_t ATTENTION_MASK = 0xF0E2;

typedef union {
  float    fval;
  uint32_t bits;
  struct {
    uint32_t frac : 23; // mantissa
    uint32_t exp  : 8;  // exponent
    uint32_t sign : 1;  // sign
  } format;
} fp32;

static inline uint32_t fp16_ieee_to_fp32_bits(uint16_t h) {
	const uint32_t w = (uint32_t) h << 16;
	const uint32_t sign = w & UINT32_C(0x80000000);
	const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
#ifdef _MSC_VER
	unsigned long nonsign_bsr;
	_BitScanReverse(&nonsign_bsr, (unsigned long) nonsign);
	uint32_t renorm_shift = (uint32_t) nonsign_bsr ^ 31;
#else
	uint32_t renorm_shift = __builtin_clz(nonsign);
#endif
	renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
	const int32_t inf_nan_mask = ((int32_t) (nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
	const int32_t zero_mask = (int32_t) (nonsign - 1) >> 31;
	return sign | ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) | inf_nan_mask) & ~zero_mask);
}

class ChatGLM {
public:
  void init(const std::vector<int> &devid, std::string model_path, std::string tokenizer_path);
  void chat();
  void deinit();
  void answer(const std::string &input_str);
  std::string predict_option(const std::string &input_str);

private:
  void tokenizer_encode(const std::string &input_str, std::vector<int> &tokens);
  int forward_first(std::vector<int> &tokens);
  std::vector<uint16_t> forward_first_without_topk(std::vector<int> &tokens); 
  int forward_next(int cur_token);
  void move2end(const bm_tensor_t &kv);
  void load_sentencepiece(std::string tokenizer_path);
  void build_system_prompt();

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  sentencepiece::SentencePieceProcessor sentencepiece;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  std::vector<bm_tensor_t> inputs_embed_512, outputs_embed_512;
  std::vector<bm_tensor_t> inputs_pid, next_pid, inputs_attention, next_attention;
  std::vector<std::vector<bm_tensor_t>> past_key, past_value;
  std::vector<bm_tensor_t> inputs_lm, outputs_lm;
  std::string name_embed;
  std::string name_embed_cache;
  std::string name_lm;
  std::vector<std::string> name_blocks;
  std::vector<std::string> name_blocks_cache;
  std::string system_string =
      "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow "
      "the user's instructions carefully. Respond using markdown.";
  std::vector<int> history_tokens;
  std::vector<int> head_prompt{64790, 64792, 64794, 30910,
                               13}; // head + system id + \n
  std::vector<int> system_prompt;
  std::vector<int> option_prompt{316, 347, 319, 367}; // A B C D

  std::map<int, std::string> option_map {
      {0, "A"},
      {1, "B"},
      {2, "C"},
      {3, "D"}
  };

  int device_num;
  int round = 0;
  int token_length;
  int EOS;
  int SEQLEN;
  int NUM_LAYERS;
};

void ChatGLM::load_sentencepiece(std::string tokenizer_path) {
  printf("Load %s ... ", tokenizer_path.c_str());
  auto status = sentencepiece.Load(tokenizer_path);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(-1);
  }
  EOS = sentencepiece.eos_id();
  printf("Done!\n");
}

void ChatGLM::init(const std::vector<int> &devices, std::string model_path, std::string tokenizer_path) {
  device_num = devices.size();
  load_sentencepiece(tokenizer_path);
  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  for (auto d : devices) {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles.push_back(h);
  }
  bm_handle = handles[0];

  // decode system prompt
  sentencepiece.Encode(system_string, &system_prompt);

  // create bmruntime
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // set NUM_LAYERS
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 2) / 2;

  // net names
  name_embed = "embedding";
  name_embed_cache = "embedding_cache";
  name_lm = "lm_head";
  for (int i = 0; i < NUM_LAYERS; i++) {
    name_blocks.emplace_back("block_" + std::to_string(i));
    name_blocks_cache.emplace_back("block_cache_" + std::to_string(i));
  }

  // net infos
  net_embed = bmrt_get_network_info(p_bmrt, name_embed.c_str());
  net_embed_cache = bmrt_get_network_info(p_bmrt, name_embed_cache.c_str());
  net_lm = bmrt_get_network_info(p_bmrt, name_lm.c_str());
  for (int i = 0; i < NUM_LAYERS; i++) {
    net_blocks.emplace_back(
        bmrt_get_network_info(p_bmrt, name_blocks[i].c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, name_blocks_cache[i].c_str()));
  }

  // set SEQLEN
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1];

  // resize
  net_blocks.resize(NUM_LAYERS);
  net_blocks_cache.resize(NUM_LAYERS);
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);

  // net device mem
  inputs_embed_512.resize(net_embed->input_num);
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&inputs_embed_512[i], p_bmrt,
                        net_embed->input_loc_devices[i],
                        net_embed->input_dtypes[i],
                        net_embed->stages[0].input_shapes[i]);
    assert(true == ret);
  }

  outputs_embed_512.resize(net_embed->output_num);
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&outputs_embed_512[i], p_bmrt,
                        net_embed->output_loc_devices[i],
                        net_embed->output_dtypes[i],
                        net_embed->stages[0].output_shapes[i]);
    assert(true == ret);
  }

  inputs_pid.resize(device_num);
  inputs_attention.resize(device_num);
  int in_num = net_blocks[0]->input_num / device_num;
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&inputs_pid[i], p_bmrt,
                        net_blocks[0]->input_loc_devices[1 + i * in_num],
                        net_blocks[0]->input_dtypes[1 + i * in_num],
                        net_blocks[0]->stages[0].input_shapes[1 + i * in_num]);
    assert(true == ret);

    ret = bmrt_tensor_ex(&inputs_attention[i], p_bmrt,
                        net_blocks[0]->input_loc_devices[2 + i * in_num],
                        net_blocks[0]->input_dtypes[2 + i * in_num],
                        net_blocks[0]->stages[0].input_shapes[2 + i * in_num]);
    assert(true == ret);
  }


  next_pid.resize(device_num);
  next_attention.resize(device_num);
  int in_num_cache = net_blocks_cache[0]->input_num / device_num;
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&next_pid[i], p_bmrt,
                        net_blocks_cache[0]->input_loc_devices[1 + i * in_num_cache],
                        net_blocks_cache[0]->input_dtypes[1 + i * in_num_cache],
                        net_blocks_cache[0]->stages[0].input_shapes[1 + i * in_num_cache]);
    assert(true == ret);

    ret = bmrt_tensor_ex(&next_attention[i], p_bmrt,
                        net_blocks_cache[0]->input_loc_devices[2 + i * in_num_cache],
                        net_blocks_cache[0]->input_dtypes[2 + i * in_num_cache],
                        net_blocks_cache[0]->stages[0].input_shapes[2 + i * in_num_cache]);
    assert(true == ret);
  }

  int out_num = net_blocks[0]->output_num / device_num;
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i].resize(device_num);
    past_value[i].resize(device_num);
    for (int j = 0; j < device_num; j++) {
      ret = bmrt_tensor_ex(&past_key[i][j], p_bmrt,
                          net_blocks[0]->output_loc_devices[1 + j * out_num],
                          net_blocks[0]->output_dtypes[1 + j * out_num],
                          net_blocks[0]->stages[0].output_shapes[1 + j * out_num]);
      assert(true == ret);
      ret = bmrt_tensor_ex(&past_value[i][j], p_bmrt,
                          net_blocks[0]->output_loc_devices[2 + j * out_num],
                          net_blocks[0]->output_dtypes[2 + j * out_num],
                          net_blocks[0]->stages[0].output_shapes[2 + j * out_num]);
      assert(true == ret);
    }
  }

  inputs_lm.resize(device_num);
  outputs_lm.resize(device_num);
  for (int i = 0; i < device_num; ++i) {
    ret = bmrt_tensor_ex(&inputs_lm[i], p_bmrt, i, net_lm->input_dtypes[0],
                        net_lm->stages[0].input_shapes[0]);
    assert(true == ret);
    ret = bmrt_tensor_ex(&outputs_lm[i], p_bmrt, i, net_lm->output_dtypes[0],
                        net_lm->stages[0].output_shapes[0]);
    assert(true == ret);
  }
}

void ChatGLM::deinit() {
  for (int i = 0; i < device_num; ++i) {
    bm_free_device(handles[i], inputs_embed_512[i].device_mem);
    bm_free_device(handles[i], outputs_embed_512[i].device_mem);
    bm_free_device(handles[i], inputs_pid[i].device_mem);
    bm_free_device(handles[i], next_pid[i].device_mem);
    bm_free_device(handles[i], inputs_attention[i].device_mem);
    bm_free_device(handles[i], next_attention[i].device_mem);
    bm_free_device(handles[i], inputs_lm[i].device_mem);
    bm_free_device(handles[i], outputs_lm[i].device_mem);
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; j++) {
      bm_free_device(handles[j], past_key[i][j].device_mem);
      bm_free_device(handles[j], past_value[i][j].device_mem);
    }
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

// after first block, move real result to end of mem
void ChatGLM::move2end(const bm_tensor_t &kv) {
  if (token_length >= SEQLEN) {
    return;
  }
  auto total_size = bm_mem_get_device_size(kv.device_mem);
  auto bytes = total_size / SEQLEN;
  auto real_size = token_length * bytes;
  auto mem =
      bm_mem_from_device(bm_mem_get_device_addr(kv.device_mem), real_size);
  auto buffer = new uint8_t[real_size];
  auto dst = new uint8_t[total_size];
  bm_memcpy_d2s(bm_handle, (void *)buffer, mem);
  memset(dst, 0, total_size - real_size);
  memcpy(dst + total_size - real_size, buffer, real_size);
  bm_memcpy_s2d(bm_handle, kv.device_mem, (void *)dst);
  delete[] buffer;
  delete[] dst;
}

int ChatGLM::forward_first(std::vector<int> &tokens) {
  std::vector<int> input_ids(SEQLEN, 0);
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, 0);

  std::copy(tokens.begin(), tokens.end(), input_ids.data());

  token_length = tokens.size();
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < SEQLEN; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i && i < token_length) {
      } else {
        attention_mask[i * SEQLEN + j] = ATTENTION_MASK;
      }
    }
  }

  // forward embeding
  std::vector<int> input_nums(device_num, 1);
  std::vector<void*> datas(device_num, (void*)input_ids.data());
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_embed_512.data(), datas.data(),
                          input_nums.data(), device_num);
  auto ret =
      bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(),
                            inputs_embed_512.data(), inputs_embed_512.size(),
                            outputs_embed_512.data(), outputs_embed_512.size(),
                            true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  std::vector<void*> pos_id_datas(device_num, position_id.data());
  std::vector<void*> in_attn_datas(device_num, attention_mask.data());
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_pid.data(), pos_id_datas.data(),
                          input_nums.data(), device_num);
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_attention.data(),in_attn_datas.data(),
                          input_nums.data(), device_num);
  auto embed_512 = outputs_embed_512;
  std::vector<bm_tensor_t> inputs_block;
  std::vector<bm_tensor_t> outputs_block;
  for (int i = 0; i < device_num; ++i) {
    embed_512[i].shape = net_blocks[0]->stages[0].input_shapes[0];
    inputs_block.push_back(embed_512[i]);
    inputs_block.push_back(inputs_pid[i]);
    inputs_block.push_back(inputs_attention[i]);
    outputs_block.push_back(embed_512[i]);
    outputs_block.push_back(past_key[0][i]);
    outputs_block.push_back(past_value[0][i]);
  }

  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; ++j) {
      outputs_block[1 + j * 3] = past_key[i][j];
      outputs_block[2 + j * 3] = past_value[i][j];
    }
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks[i].c_str(),
                                inputs_block.data(), inputs_block.size(),
                                outputs_block.data(), outputs_block.size(),
                                true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
    for (int j = 0; j < device_num; ++j) {
      move2end(past_key[i][j]);
      move2end(past_value[i][j]);
    }
  }

  // forward lmhead
  int bytes = embed_512[0].device_mem.size / SEQLEN;
  bm_memcpy_d2d_byte(bm_handle, inputs_lm[0].device_mem, 0,
                     embed_512[0].device_mem, (token_length - 1) * bytes,
                     bytes);
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm[0], 1,
                              &outputs_lm[0], 1, true, false);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm[0].device_mem);
  return token;
}

int ChatGLM::forward_next(int cur_token) {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = 0; i <= SEQLEN - token_length; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;

  // forward embedding
  std::vector<bm_tensor_t> inputs_embed;
  std::vector<void*> input_datas;
  std::vector<int> input_nums(device_num, 1);
  for (int i = 0; i < device_num; ++i) {
    inputs_embed.push_back(outputs_lm[i]); // token_id
    inputs_embed[i].shape = net_embed_cache->stages[0].input_shapes[0];
    input_datas.push_back((void*)(&cur_token));
  }
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_embed.data(), input_datas.data(),
                          input_nums.data(), device_num);
  auto ret = bmrt_launch_tensor_ex(p_bmrt, name_embed_cache.c_str(),
                                  inputs_embed.data(), inputs_embed.size(),
                                  inputs_lm.data(), inputs_lm.size(), true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  std::vector<void*> attn_datas(device_num, attention_mask.data());
  std::vector<void*> pid_datas(device_num, &position_id);
  bmrt_memcpy_s2d_parallel(p_bmrt, next_attention.data(), attn_datas.data(),
                          input_nums.data(), device_num);
  bmrt_memcpy_s2d_parallel(p_bmrt, next_pid.data(), pid_datas.data(),
                          input_nums.data(), device_num);
                          
  // WARNING: make inputs_lm device_num                   
  std::vector<bm_tensor_t> embed_1 = inputs_lm;
  for (int i = 0; i < device_num; ++i) {
    embed_1[i].shape = net_blocks_cache[0]->stages[0].input_shapes[0];
  }
  std::vector<bm_tensor_t> inputs_block;
  std::vector<bm_tensor_t> outputs_block;
  for (int i = 0; i < device_num; ++i) {
    inputs_block.push_back(embed_1[i]);
    inputs_block.push_back(next_pid[i]);
    inputs_block.push_back(next_attention[i]);
    inputs_block.push_back(past_key[0][i]);
    inputs_block.push_back(past_value[0][i]);
    outputs_block.push_back(embed_1[i]);
    outputs_block.push_back(past_key[0][i]);
    outputs_block.push_back(past_value[0][i]);
  }

  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; ++j) {
      inputs_block[3 + j * 5] = past_key[i][j];
      inputs_block[4 + j * 5] = past_value[i][j];
      outputs_block[1 + j * 3] = past_key[i][j];
      outputs_block[2 + j * 3] = past_value[i][j];
    }
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks_cache[i].c_str(),
                                inputs_block.data(), inputs_block.size(),
                                outputs_block.data(), outputs_block.size(),
                                true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
  }

  // forward lmhead
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm[0], 1,
                              &outputs_lm[0], 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm[0].device_mem);
  return token;
}

void ChatGLM::build_system_prompt() {
  history_tokens.clear();
  history_tokens.insert(history_tokens.end(), head_prompt.begin(),
                        head_prompt.end());
  history_tokens.insert(history_tokens.end(), system_prompt.begin(),
                        system_prompt.end());
}

void ChatGLM::chat() {
  while (true) {
    std::cout << "\nQuestion: ";
    std::string input_str;
    std::getline(std::cin, input_str);
    if (input_str == "exit") {
      break;
    }
    std::cout << "\nAnswer: " << std::flush;
    answer(input_str);
    std::cout << std::endl;
  }
}

void ChatGLM::answer(const std::string &input_str) {
  // auto time_0 = std::chrono::system_clock::now();
  int tok_num = 0;
  std::vector<int> tokens;
  std::vector<int> prompt{64795, 30910, 13};
  sentencepiece.Encode(input_str, &tokens);
  tokens.insert(tokens.begin(), prompt.begin(), prompt.end());
  tokens.push_back(64796);
  if (history_tokens.size() == 0) {
    build_system_prompt();
  }
  history_tokens.insert(history_tokens.end(), tokens.begin(), tokens.end());
  
  if (history_tokens.empty()) {
    printf("Sorry: your question is too wierd!!\n");
    round = 0;
    history_tokens.clear();
    return;
  }
  // make sure token not too large
  if ((int)history_tokens.size() > SEQLEN - 10) {
    // reset
    history_tokens.clear();
    if (round == 0) {
      printf("Error: your question is too large!\n");
      return;
    }
    round = 0;
    answer(input_str);
    return;
  }
  int pre_token = 0;
  auto t0 = std::chrono::system_clock::now();
  int token = forward_first(history_tokens);
  auto t1 = std::chrono::system_clock::now();
  while (token != EOS && token_length < SEQLEN) {
    std::string pre_word;
    std::string word;
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    sentencepiece.Decode(pre_ids, &pre_word);
    sentencepiece.Decode(ids, &word);
    std::string diff = word.substr(pre_word.size());
    history_tokens.emplace_back(token);
    std::cout << diff << std::flush;
    if (token_length < SEQLEN) {
      token_length++;
    }
    tok_num++;
    token = forward_next(token);
  }
  auto t2 = std::chrono::system_clock::now();
  auto use0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  auto use1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  printf("\n\nfirst token latency: %f s", (use0.count() * 1e-6));
  printf("\nspeed: %f token/s\n", tok_num / (use1.count() * 1e-6));
  if (token_length >= SEQLEN) {
    history_tokens.clear();
    round = 0;
  } else {
    round++;
  }
}

// for c-eval
std::vector<uint16_t> ChatGLM::forward_first_without_topk(std::vector<int> &tokens) {
  std::vector<int> input_ids(SEQLEN, 0);
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, 0);

  std::copy(tokens.begin(), tokens.end(), input_ids.data());

  token_length = tokens.size();
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < SEQLEN; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i && i < token_length) {
      } else {
        attention_mask[i * SEQLEN + j] = ATTENTION_MASK;
      }
    }
  }

  // forward embeding
  std::vector<int> input_nums(device_num, 1);
  std::vector<void*> datas(device_num, (void*)input_ids.data());
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_embed_512.data(), datas.data(),
                          input_nums.data(), device_num);
  auto ret =
      bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(),
                            inputs_embed_512.data(), inputs_embed_512.size(),
                            outputs_embed_512.data(), outputs_embed_512.size(),
                            true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  std::vector<void*> pos_id_datas(device_num, position_id.data());
  std::vector<void*> in_attn_datas(device_num, attention_mask.data());
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_pid.data(), pos_id_datas.data(),
                          input_nums.data(), device_num);
  bmrt_memcpy_s2d_parallel(p_bmrt, inputs_attention.data(),in_attn_datas.data(),
                          input_nums.data(), device_num);
  auto embed_512 = outputs_embed_512;
  std::vector<bm_tensor_t> inputs_block;
  std::vector<bm_tensor_t> outputs_block;
  for (int i = 0; i < device_num; ++i) {
    embed_512[i].shape = net_blocks[0]->stages[0].input_shapes[0];
    inputs_block.push_back(embed_512[i]);
    inputs_block.push_back(inputs_pid[i]);
    inputs_block.push_back(inputs_attention[i]);
    outputs_block.push_back(embed_512[i]);
    outputs_block.push_back(past_key[0][i]);
    outputs_block.push_back(past_value[0][i]);
  }

  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; ++j) {
      outputs_block[1 + j * 3] = past_key[i][j];
      outputs_block[2 + j * 3] = past_value[i][j];
    }
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks[i].c_str(),
                                inputs_block.data(), inputs_block.size(),
                                outputs_block.data(), outputs_block.size(),
                                true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
    for (int j = 0; j < device_num; ++j) {
      move2end(past_key[i][j]);
      move2end(past_value[i][j]);
    }
  }

  // forward lmhead
  int bytes = embed_512[0].device_mem.size / SEQLEN;
  bm_memcpy_d2d_byte(bm_handle, inputs_lm[0].device_mem, 0,
                     embed_512[0].device_mem, (token_length - 1) * bytes,
                     bytes);
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm[0], 1,
                              &outputs_lm[0], 1, true, false);
  bm_thread_sync(bm_handle);

  int vocab_size = net_lm->stages[0].output_shapes[0].dims[1];
  std::vector<uint16_t> logits(vocab_size);
  bm_memcpy_d2s(bm_handle, logits.data(), outputs_lm[0].device_mem);
  return logits;
}

std::string ChatGLM::predict_option(const std::string &input_str) {
  int tok_num = 0;
  std::vector<int> tokens;
  sentencepiece.Encode(input_str, &tokens);

  std::vector<int> user_prompt{64795, 30910, 13};
  tokens.insert(tokens.begin(), user_prompt.begin(), user_prompt.end());

  std::vector<int> assistant_prompt{64796, 30910, 13};
  tokens.insert(tokens.end(), assistant_prompt.begin(), assistant_prompt.end());

  build_system_prompt();
  history_tokens.insert(history_tokens.end(), tokens.begin(), tokens.end());
  
  if (history_tokens.empty()) {
    history_tokens.clear();
    printf("Sorry: your question is too wierd!!\n");
    return {};
  }
  // make sure token not too large
  if ((int)history_tokens.size() > SEQLEN - 10) {
    // reset
    std::cout << "Token Size : " << history_tokens.size() << std::endl;
    history_tokens.clear();
    printf("Error: your question is too large!\n");
    return {};
  }

  std::vector<uint16_t> logits = forward_first_without_topk(history_tokens);
  if (token_length >= SEQLEN) {
    history_tokens.clear();
  }

  // convert fp16 to fp32
  std::vector<float> fp32_logits;
  for(int i = 0; i < option_prompt.size(); i++){
    fp32 t;
    t.bits = fp16_ieee_to_fp32_bits(logits[option_prompt[i]]);
    fp32_logits.push_back(t.fval);
  }

  // get the index of maximum and map the index to option_map
  auto max_it = std::max_element(fp32_logits.begin(), fp32_logits.end());
  int max_index = std::distance(fp32_logits.begin(), max_it);

  return option_map[max_index];
}

PYBIND11_MODULE(chat, m) {
    pybind11::class_<ChatGLM>(m, "ChatGLM")
        .def(pybind11::init<>())
        .def("init", &ChatGLM::init)
        .def("answer", &ChatGLM::answer)
        .def("predict_option", &ChatGLM::predict_option)
        .def("deinit", &ChatGLM::deinit);
}
