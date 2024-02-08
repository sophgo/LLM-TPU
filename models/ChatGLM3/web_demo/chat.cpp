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

static const int NUM_LAYERS = 28;
static const int MAX_LEN = 512;
static const int HIDDEN_SIZE = 4096;

static const std::string TOKENIZER_MODEL = "tokenizer.model";
static const std::string CHATGLM_MODEL = "chatglm3-6b.bmodel";

struct TokenWord{
  int token;
  std::string word;
};

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

class ChatGLM {
public:
  ChatGLM(){}
  void init(int devid);
  void init(int devid, const std::string bmodel_path, const std::string tokenizer_path);
  // void chat();
  std::string answer(const std::string &input_str);
  void deinit();
  std::string name;
  std::string history = "";
  int round = 0;
  int token_length;
  int EOS;
  std::string predict_next_token();
  std::string predict_first_token(const std::string &input_str);
  void build_system_prompt();

private:
  void tokenizer_encode(const std::string &input_str, std::vector<int> &tokens);
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  void move2end(const bm_tensor_t &kv);
  void load_sentencepiece();
  void load_sentencepiece(const std::string &tokenizer_path);
  std::string system_string = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.";
  std::vector<int> history_tokens;
  std::vector<int> head_prompt{64790, 64792, 64794, 30910, 13}; // head + system id + \n
  std::vector<int> system_prompt;


private:
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
  std::string name_embed;
  std::string name_lm;
  std::string name_blocks[NUM_LAYERS];
  std::string name_blocks_cache[NUM_LAYERS];
};

void ChatGLM::load_sentencepiece() {
  printf("Load %s ... ", TOKENIZER_MODEL.c_str());
  auto status = sentencepiece.Load(TOKENIZER_MODEL);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(-1);
  }
  EOS = sentencepiece.eos_id();
  printf("Done!\n");
}

void ChatGLM::load_sentencepiece(const std::string &model) {
  printf("Load %s ... ", model.c_str());
  auto status = sentencepiece.Load(model);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(-1);
  }
  EOS = sentencepiece.eos_id();
  printf("Done!\n");
}

void ChatGLM::init(int devid, const std::string bmodel_path, const std::string tokenizer_path){
  load_sentencepiece(tokenizer_path);
  bm_status_t status = bm_dev_request(&bm_handle, devid);
  assert(BM_SUCCESS == status);

  // decode system prompt
  sentencepiece.Encode(system_string, &system_prompt);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", bmodel_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, bmodel_path.c_str());
  assert(true == ret);
  printf("Done!\n");
  // net names
  name_embed = "embedding";
  name_lm = "lm_head";
  for (int i = 0; i < NUM_LAYERS; i++) {
    name_blocks[i] = "glm_block_" + std::to_string(i);
    name_blocks_cache[i] = "glm_block_cache_" + std::to_string(i);
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
  }
  ret = bmrt_tensor(&inputs_lm, p_bmrt, net_lm->input_dtypes[0],
                    net_lm->stages[0].input_shapes[0]);
  assert(true == ret);
  ret = bmrt_tensor(&outputs_lm, p_bmrt, net_lm->output_dtypes[0],
                    net_lm->stages[0].output_shapes[0]);
  assert(true == ret);
}


void ChatGLM::init(int devid) {
  load_sentencepiece();
  // request bm_handle
  bm_status_t status = bm_dev_request(&bm_handle, devid);
  assert(BM_SUCCESS == status);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", CHATGLM_MODEL.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, CHATGLM_MODEL.c_str());
  assert(true == ret);
  printf("Done!\n");
  // net names
  name_embed = "embedding";
  name_lm = "lm_head";
  for (int i = 0; i < NUM_LAYERS; i++) {
    name_blocks[i] = "glm_block_" + std::to_string(i);
    name_blocks_cache[i] = "glm_block_cache_" + std::to_string(i);
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
  }
  ret = bmrt_tensor(&inputs_lm, p_bmrt, net_lm->input_dtypes[0],
                    net_lm->stages[0].input_shapes[0]);
  assert(true == ret);
  ret = bmrt_tensor(&outputs_lm, p_bmrt, net_lm->output_dtypes[0],
                    net_lm->stages[0].output_shapes[0]);
  assert(true == ret);
}

void ChatGLM::deinit() {
  bm_free_device(bm_handle, inputs_embed_512.device_mem);
  bm_free_device(bm_handle, outputs_embed_512.device_mem);
  bm_free_device(bm_handle, inputs_lm.device_mem);
  bm_free_device(bm_handle, outputs_lm.device_mem);
  bm_free_device(bm_handle, inputs_pid.device_mem);
  bm_free_device(bm_handle, next_pid.device_mem);
  bm_free_device(bm_handle, inputs_attention.device_mem);
  bm_free_device(bm_handle, next_attention.device_mem);
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_free_device(bm_handle, past_key[i].device_mem);
    bm_free_device(bm_handle, past_value[i].device_mem);
  }
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

// after first block, move real result to end of mem
void ChatGLM::move2end(const bm_tensor_t &kv) {
  if (token_length >= MAX_LEN) {
    return;
  }
  auto total_size = bm_mem_get_device_size(kv.device_mem);
  auto bytes = total_size / MAX_LEN;
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
  std::vector<int> input_ids(MAX_LEN, 0);
  std::vector<int> position_id(MAX_LEN, 0);
  std::vector<float> attention_mask(MAX_LEN * MAX_LEN, 0);

  input_ids[0] = 64790;
  input_ids[1] = 64792;
  std::copy(tokens.begin(), tokens.end(), input_ids.data() + 2);

  token_length = tokens.size() + 2;
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < MAX_LEN; i++) {
    for (int j = 0; j < MAX_LEN; j++) {
      if (j <= i && i < token_length) {
      } else {
        attention_mask[i * MAX_LEN + j] = 1.0;
      }
    }
  }

  // forward embeding
  bm_memcpy_s2d(bm_handle, inputs_embed_512.device_mem, (void *)input_ids.data());
  auto ret =
      bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(), &inputs_embed_512, 1,
                            &outputs_embed_512, 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);

  // forward blocks
  bm_memcpy_s2d(bm_handle, inputs_pid.device_mem, (void *)position_id.data());
  bm_memcpy_s2d(bm_handle, inputs_attention.device_mem, (void *)attention_mask.data());
  auto inputs_embed = outputs_embed_512;
  inputs_embed.shape = net_blocks[0]->stages[0].input_shapes[0];
  bm_tensor_t inputs_block[3] = {inputs_embed, inputs_pid, inputs_attention};
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_tensor_t outputs_block[3] = {inputs_embed, past_key[i], past_value[i]};
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks[i].c_str(), inputs_block, 3,
                                outputs_block, 3, true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
    move2end(past_key[i]);
    move2end(past_value[i]);
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

int ChatGLM::forward_next() {
  std::vector<float> attention_mask(MAX_LEN + 1, 0);
  for (int i = 0; i <= MAX_LEN - token_length; i++) {
    attention_mask[i] = 1.0;
  }
  int32_t position_id = token_length - 1;
  // embedding
  outputs_lm.shape = net_embed->stages[0].input_shapes[0];
  auto ret = bmrt_launch_tensor_ex(p_bmrt, name_embed.c_str(), &outputs_lm, 1,
                                   &inputs_lm, 1, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
  // blocks
  bm_memcpy_s2d(bm_handle, next_attention.device_mem, (void *)attention_mask.data());
  bm_memcpy_s2d(bm_handle, next_pid.device_mem, (void *)&position_id);
  auto inputs_embed = inputs_lm;
  inputs_embed.shape = net_blocks_cache[0]->stages[0].input_shapes[0];
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_tensor_t inputs_block[5] = {inputs_embed, next_pid, next_attention,
                                   past_key[i], past_value[i]};
    bm_tensor_t outputs_block[3] = {inputs_embed, past_key[i], past_value[i]};
    ret = bmrt_launch_tensor_ex(p_bmrt, name_blocks_cache[i].c_str(),
                                inputs_block, 5, outputs_block, 3, true, false);
    assert(ret);
    bm_thread_sync(bm_handle);
  }
  outputs_lm.shape = net_lm->stages[0].output_shapes[0];
  ret = bmrt_launch_tensor_ex(p_bmrt, name_lm.c_str(), &inputs_lm, 1,
                              &outputs_lm, 1, true, false);
  bm_thread_sync(bm_handle);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, outputs_lm.device_mem);
  return token;
}

void ChatGLM::build_system_prompt() {
  history_tokens.clear();
  history_tokens.insert(history_tokens.end(), head_prompt.begin(), head_prompt.end());
  history_tokens.insert(history_tokens.end(), system_prompt.begin(), system_prompt.end());
}

std::string ChatGLM::predict_first_token(const std::string &input_str) {
  //int tok_num = 1;
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
    round = 0;
    history = "Sorry: your question is too wierd!!\n";
    history_tokens.clear();
    return history;
  }
  // make sure token not too large
  if (history_tokens.size() > MAX_LEN - 10) {
    // reset
    history_tokens.clear();
    round = 0;
    return predict_first_token(input_str);
  }
  int token = forward_first(history_tokens);
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
  history_tokens.emplace_back(token);
  if (token_length < MAX_LEN) {
    token_length++;
  }
  return diff;
}

std::string ChatGLM::predict_next_token() {
  int pre_token;
  pre_token = 0;
  int token = forward_next();
  if(token == EOS){
    round = 0;
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
  history_tokens.emplace_back(token);
  if (token_length < MAX_LEN) {
    token_length++;
  }else{
    round = 0;
    history_tokens.clear();
    return "_GETMAX_";
  }
  return diff;
}


extern "C" {


ChatGLM *ChatGLM_with_devid_and_model(int devid, const char *bmodel_path, const char *tokenizer_path) {
  ChatGLM *chat = new ChatGLM();
  chat->init(devid, bmodel_path, tokenizer_path);
  return chat;
}

void ChatGLM_delete(ChatGLM *chat) { delete chat; }

void ChatGLM_deinit(ChatGLM *chat) {
  chat->deinit();
}

const char *get_history(ChatGLM *chat) {
  std::string str = chat->history;
  return strdup(str.c_str());
}

const char *set_history(ChatGLM *chat, const char *history) {
  chat->history = history;
  return strdup(history);
}

const char *ChatGLM_predict_first_token(ChatGLM *chat, const char *input_str) {
  std::string str = chat->predict_first_token(input_str);
  return strdup(str.c_str());
}

const char *ChatGLM_predict_next_token(ChatGLM *chat) {
  std::string str = chat->predict_next_token();
  return strdup(str.c_str());
}

const int get_eos(ChatGLM *chat){
  const int res = chat->EOS;
  return res;
}
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <bmodel_path> <tokenizer_path>" << std::endl;
    return -1;
  }
  const char* bmodel_path = argv[1];
  const char* tokenizer_path = argv[2];
  ChatGLM *chat = ChatGLM_with_devid_and_model(0, bmodel_path, tokenizer_path);
  const char *inputstr = "请用python写排序";
  const char *outputstr = ChatGLM_predict_first_token(chat, inputstr);
  std::cout << outputstr << std::flush;
  for (int i = 0; i < 10; i++) {
    outputstr = ChatGLM_predict_next_token(chat);
    std::cout << outputstr << std::flush;
  }
  return 0;
}
