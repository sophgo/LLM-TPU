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

static const uint16_t ATTENTION_MASK = 0xF0E2;

class LLama2 {
public:
  void init(const std::vector<int> &devid, std::string model_path, std::string tokenizer_path);
  void chat();
  void deinit();

private:
  void answer(const std::string &input_str);
  void tokenizer_encode(const std::string &input_str, std::vector<int> &tokens);
  int forward_first(std::vector<int> &tokens);
  int forward_next(int cur_token);
  void load_sentencepiece(std::string tokenizer_path);

private:
  int device_num;
  bm_handle_t bm_handle;
  std::vector<bm_handle_t> handles;
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
  std::vector<bm_tensor_t> present_key_cache, present_value_cache;
  std::vector<bm_tensor_t> inputs_lm, outputs_lm;
  std::string history = "";
  std::string name_embed;
  std::string name_embed_cache;
  std::string name_lm;
  std::vector<std::string> name_blocks;
  std::vector<std::string> name_blocks_cache;
  int SEQLEN;     // read from bmodel
  int NUM_LAYERS; // read from bmodel
  int round = 0;
  int token_length;
  int EOS;
};

void LLama2::load_sentencepiece(std::string tokenizer_path) {
  printf("Load %s ... ", tokenizer_path.c_str());
  auto status = sentencepiece.Load(tokenizer_path);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(-1);
  }
  EOS = sentencepiece.eos_id();
  printf("Done!\n");
}

void LLama2::init(const std::vector<int> &devices, std::string model, std::string tokenizer_path) {
  // load tokenizer
  load_sentencepiece(tokenizer_path);

  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  device_num = devices.size();
  for (auto d : devices) {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles.push_back(h);
  }
  bm_handle = handles[0];

  // create bmruntime
  p_bmrt = bmrt_create_ex(handles.data(), device_num);
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model.c_str());
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

  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
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

  present_key_cache.resize(device_num);
  present_value_cache.resize(device_num);
  inputs_lm.resize(device_num);
  outputs_lm.resize(device_num);
  for (int i = 0; i < device_num; ++i) {
    present_key_cache[i] = past_key[0][i];
    present_value_cache[i] = past_value[0][i];
    present_key_cache[i].shape.dims[1] = 1;
    present_value_cache[i].shape.dims[1] = 1;

    ret = bmrt_tensor_ex(&inputs_lm[i], p_bmrt, i, net_lm->input_dtypes[0],
                        net_lm->stages[0].input_shapes[0]);
    assert(true == ret);
    ret = bmrt_tensor_ex(&outputs_lm[i], p_bmrt, i, net_lm->output_dtypes[0],
                        net_lm->stages[0].output_shapes[0]);
    assert(true == ret);
  }
}

void LLama2::deinit() {
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

int LLama2::forward_first(std::vector<int> &tokens) {
  // make inputs
  std::vector<int> input_ids(SEQLEN, 0);
  std::vector<int> position_id(SEQLEN, 0);  
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), input_ids.data());
  token_length = tokens.size();
  
  std::copy(tokens.begin(), tokens.end(), input_ids.data());
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }

  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  std::vector<int> input_nums(device_num, 1);
  std::vector<void*> datas(device_num, input_ids.data());
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
  }

  int bytes = embed_512[0].device_mem.size / SEQLEN;
  bm_memcpy_d2d_byte(bm_handle, inputs_lm[0].device_mem, 0,
                     embed_512[0].device_mem, (token_length - 1) * bytes,
                     bytes);
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm[0], 1,
                              &outputs_lm[0], 1,
                              true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
  
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm[0].device_mem);
  return token;
}

int LLama2::forward_next(int cur_token) {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;

  // embedding
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

  // blocks
  std::vector<void*> pid_datas(device_num, &position_id);
  std::vector<void*> attn_datas(device_num, attention_mask.data());
  bmrt_memcpy_s2d_parallel(p_bmrt, next_pid.data(), pid_datas.data(),
                          input_nums.data(), device_num);
  bmrt_memcpy_s2d_parallel(p_bmrt, next_attention.data(), attn_datas.data(),
                          input_nums.data(), device_num);
  std::vector<bm_tensor_t> embed_1 = inputs_lm;
  for (int i = 0; i < device_num; ++i) {
    embed_1[i].shape = net_blocks_cache[0]->stages[0].input_shapes[0];
  }
  int bytes = bm_mem_get_device_size(past_key[0][0].device_mem) / SEQLEN;
  int token_offset = (token_length - 1) * bytes;
  std::vector<bm_tensor_t> inputs_block;
  std::vector<bm_tensor_t> outputs_block;
  for (int i = 0; i < device_num; ++i) {
    inputs_block.push_back(embed_1[i]);
    inputs_block.push_back(next_pid[i]);
    inputs_block.push_back(next_attention[i]);
    inputs_block.push_back(past_key[0][i]);
    inputs_block.push_back(past_value[0][i]);
    outputs_block.push_back(embed_1[i]);
    outputs_block.push_back(present_key_cache[i]);
    outputs_block.push_back(present_value_cache[i]);
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    for (int j = 0; j < device_num; ++j) {
      inputs_block[3 + j * 5] = past_key[i][j];
      inputs_block[4 + j * 5] = past_value[i][j];
      bm_set_device_mem(&outputs_block[1 + j * 3].device_mem, bytes,
          bm_mem_get_device_addr(past_key[i][j].device_mem) + token_offset);
      bm_set_device_mem(&outputs_block[2 + j * 3].device_mem, bytes,
          bm_mem_get_device_addr(past_value[i][j].device_mem) + token_offset);
    }
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks_cache[i].c_str(),
                                inputs_block.data(), inputs_block.size(),
                                outputs_block.data(), outputs_block.size(),
                                true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
  }

  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm[0], 1,
                              &outputs_lm[0], 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm[0].device_mem);
  return token;
}

void LLama2::chat() {
  while (true) {
    std::cout << "\nQuestion: ";
    std::string input_str;
    std::getline(std::cin, input_str);
    std::string sys_config = R"(
            [INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n)";
    if (input_str == "exit") {
      break;
    }

    input_str = sys_config + input_str + " [/INST] ";
    if(history == "") {
      input_str = sys_config + "\nQuestion:\n" + input_str + "\nAnswer\n:";
    }
    else {
      input_str = "\nQuestion:\n" + input_str + "\nAnswer:\n";
    }
    std::cout << "\nAnswer: " << std::flush;
    answer(input_str);
    std::cout << std::endl;
  }
}

void LLama2::answer(const std::string &input_str) {
  history = input_str;
  int tok_num = 1;
  std::vector<int> tokens;
  sentencepiece.Encode(history, &tokens);
  tokens.insert(tokens.begin(), 1);
  if (tokens.empty()) {
    printf("Sorry: your question is too wierd!!\n");
    history = "";
    round = 0;
    return;
  }
  // make sure token not too large
  token_length = tokens.size();
  if (token_length > SEQLEN - 10) {
    // reset
    if (round == 0) {
      printf("Error: your question is too large!\n");
      return;
    }
    round = 0;
    history = "";
    answer(input_str);
    return;
  }
  int pre_token = 0;
  auto t0 = std::chrono::system_clock::now();
  int token = forward_first(tokens);
  auto t1 = std::chrono::system_clock::now();
  while (token != EOS && token_length < SEQLEN) {
    std::string pre_word;
    std::string word;
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    sentencepiece.Decode(pre_ids, &pre_word);
    sentencepiece.Decode(ids, &word);
    std::string diff = word.substr(pre_word.size());
    history += diff;
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
    round = 0;
    history = history.substr(history.size() / 2);
  } else {
    history += "\n\n";
    round++;
  }
}

static void split(const std::string &s, const std::string &delim,
                  std::vector<std::string> &ret) {
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret.push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (last < s.length()) {
    ret.push_back(s.substr(last));
  }
}

static std::vector<int> parseCascadeDevices(const std::string &str) {
  std::vector<int> devices;
  std::vector<std::string> sub_str;
  split(str, ",", sub_str);
  for (auto &s : sub_str) {
    devices.push_back(std::atoi(s.c_str()));
  }
  return devices;
}

void Usage() {
  printf("Usage:\n"
         "  --help         : Show help info.\n"
         "  --model        : Set model path \n"
         "  --tokenizer    : Set tokenizer path \n"
         "  --devid        : Set devices to run for model, e.g. 1,2. if not "
         "set, use 0\n");
}

void processArguments(int argc, char *argv[], std::string &model_path, std::string &tokenizer_path,
                      std::vector<int> &devices) {
  struct option longOptions[] = {{"model", required_argument, nullptr, 'm'},
                                 {"tokenizer", required_argument, nullptr, 't'},
                                 {"devid", required_argument, nullptr, 'd'},
                                 {"help", no_argument, nullptr, 'h'},
                                 {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:d:h:", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      model_path = optarg;
      break;
    case 't':
      tokenizer_path = optarg;
      break;
    case 'd':
      devices = parseCascadeDevices(optarg);
      break;
    case 'h':
      Usage();
      exit(EXIT_FAILURE);
    case '?':
      Usage();
      exit(EXIT_FAILURE);
    default:
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char **argv) {
  // set your bmodel path here
  printf("Demo for LLama2 in BM1684X\n");
  std::string model_path;
  std::string tokenizer_path;
  std::vector<int> devices = {0};
  processArguments(argc, argv, model_path, tokenizer_path, devices);

  LLama2 llama;
  printf("Init Environment ...\n");
  llama.init(devices, model_path, tokenizer_path);
  printf("==========================\n");
  llama.chat();
  llama.deinit();
  return 0;
}
