#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bmruntime_interface.h"
#include "memory.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <vector>

#include "bmruntime_interface.h"

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

class RWKV6 {
public:
  void init(const std::vector<int> &devices, std::string model_path);
  void init_state(); // Initialize a default prompt
  void deinit();

  // forward method
  int prefill(std::vector<uint32_t> &tokens, bool cache_state2mem = false,
              bool use_previous_state = false, bool use_cached_state = false);

  int rnn_gen(bool use_cached_state = false, bool cache_state2mem = false);

  // Normal chat interface
  std::vector<int> generate(std::vector<uint32_t> &input_tokens, int EOS);

  std::mt19937 sgen;
  RWKV6() : sgen(std::random_device()()){};

private:
  // Internal implementation
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void net_launch(const bm_net_info_t *net, uint32_t stage_idx = 0);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

public:
  int token_length; // Generated tokens id
  int SEQLEN = 4096;
  int NUM_LAYERS;
  std::string generation_mode;
  std::string prompt_mode;            // Not used for now
  int max_new_tokens;                 // for test
  std::vector<int> visited_tokens;    // Token range visible to the model
  bool state_calculated_flag = false; // Whether the Blocks output memory holds a state
  bool state_mem_cached_flag =
      false; // Whether the state cache mem holds a state (TODO: change to a state tree later)

  // generation
  float temperature;
  float top_p;
  float repeat_penalty;
  int repeat_last_n;

private:
  int device_num = 0;
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;

  // Model parameters
  int STATE_SIZE_1 = 0; // RWKV state size
  int STATE_SIZE_2 = 0; // RWKV state size
  std::vector<const bm_net_info_t *> net_blocks;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_lm_head;
  const bm_net_info_t *net_greedy_head;
  const bm_net_info_t *net_penalty_sample_head;
  // Internal variables
  const uint16_t STATE_INIT_DATA = 0x0000;
  std::vector<std::vector<uint32_t>> tokens_temp; // temp the token to be infer
  bm_device_mem_t state_cache;                    // state cache
};
/**
 * run a net
 */
void RWKV6::net_launch(const bm_net_info_t *net, uint32_t stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
  }
  bool ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  // bm_thread_sync(bm_handle);
}

/**
 * copy from device to device
 */
void RWKV6::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

/**
 * init rwkv model
 */
void RWKV6::init(const std::vector<int> &devices, std::string model_path) {

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

// create bmruntime
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");
  print_devmem_info(handles[0]);

  // get rwkv model
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_lm_head = bmrt_get_network_info(p_bmrt, "lm_head");
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  net_penalty_sample_head =
      bmrt_get_network_info(p_bmrt, "penalty_sample_head");
  SEQLEN = net_penalty_sample_head->stages[0]
               .input_shapes[1]
               .dims[1]; // TODO:
                         // Sequence length; change to unlimited later, but keep limiting the repetition length for penalty sampling
  int num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = num_nets - 4;

  // resize
  visited_tokens.resize(SEQLEN);

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    std::string block_name = "block_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
  }

  // get state size
  int state_byte_size_1 = net_blocks[0]->max_input_bytes[0];
  int state_byte_size_2 = net_blocks[0]->max_input_bytes[1];
  int first_state_size = state_byte_size_2 / state_byte_size_1;
  int second_state_size = net_embed->stages[0].output_shapes->dims[1];
  STATE_SIZE_1 = first_state_size;
  STATE_SIZE_2 = second_state_size;

  auto mem_ret = bm_malloc_device_byte(bm_handle, &state_cache,
                                       net_blocks[0]->max_input_bytes[1]);
  assert(BM_SUCCESS == mem_ret);

  return;
}
/**
 * deinit
 */
void RWKV6::deinit() {

  bm_free_device(bm_handle, state_cache);

  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

/**
 * run model head
 */
void RWKV6::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device(&in_tensors[0], logits_mem, net->input_dtypes[0],
                          net->stages[0].input_shapes[0]);

  for (int i = 1; i < net->input_num; i++) {
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
  // bm_thread_sync(bm_handle);
}
/**
 * greedy search
 */
int RWKV6::greedy_search(const bm_net_info_t *net,
                         bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}
/**
 * penalty sample
 * TODO: need to fix this
 */
int RWKV6::penalty_sample(const bm_net_info_t *net,
                          bm_device_mem_t &logits_mem) {
  auto &in1_mem = net->stages[0].input_mems[1];
  auto &in2_mem = net->stages[0].input_mems[2];
  auto &in3_mem = net->stages[0].input_mems[3];
  auto &in4_mem = net->stages[0].input_mems[4];
  auto &out0_mem = net->stages[0].output_mems[0];
  auto &out1_mem = net->stages[0].output_mems[1];

  std::vector<int> generated_tokens(
      SEQLEN, visited_tokens[token_length -
                             1]); // SEQLEN is the sampler's length limit; change to unlimited later
  repeat_last_n =
      std::min(repeat_last_n, token_length); // Total generated token length or repetition detection range
  std::copy(visited_tokens.begin() + token_length - repeat_last_n,
            visited_tokens.begin() + token_length, generated_tokens.begin());
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)generated_tokens.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, in4_mem, (void *)&repeat_penalty);

  // inference
  head_launch(net, logits_mem);

  // get logit & token
  int candidate_num = net->stages[0].output_shapes[0].dims[1];
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s(bm_handle, probs.data(), out0_mem);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s(bm_handle, tokens.data(), out1_mem);

  // penalty_sample
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

/**
 * @brief Prefill: input a sequence of tokens, fill and generate the current state, placed in the model output memory.
 * If @param cache_state2mem is enabled, also copy a copy into the state cache tensor.
 * TODO:
 * Expand the state cache slots to 50 (exact number TBD) plus a management system mapping states to text (expected to be a Trie)
 *
 * To continue filling from a previous state, enable @param use_previous_state
 *
 * @param tokens
 * @param cache_state2mem
 * @param use_previous_state  Use the state in the model memory
 * @param use_cached_state    Use the actively cached state
 * @return int
 */
int RWKV6::prefill(std::vector<uint32_t> &tokens, bool cache_state2mem,
                   bool use_previous_state, bool use_cached_state) {
  token_length = tokens.size();
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

  if (use_previous_state)
    assert(state_calculated_flag ==
           true); // Using the in-model state without a prior prefill is forbidden
  if (use_cached_state)
    assert(state_mem_cached_flag ==
           true); // Using the cached state without having cached one is forbidden

  assert((use_previous_state && use_cached_state) ==
         0); // Using both caches at the same time is forbidden!
  std::vector<uint16_t> state_init_data(STATE_SIZE_1 * STATE_SIZE_2,
                                        STATE_INIT_DATA);
  /**
   * @brief State usage logic
   *
   * 0  Not using the state in model memory + not using the cached state = use empty state
   * 1  Using the state in model memory + not using the cached state   = use in-memory state
   * 2  Not using the state in model memory + using the cached state   = use actively cached state
   *
   * The two states cannot be used at the same time!!
   */
  // int state_logic = 0;
  // if (!use_previous_state && !use_cached_state)
  //   state_logic = 0;
  // else if (use_previous_state && !use_cached_state)
  //   state_logic = 1;
  // else if (!use_previous_state && use_cached_state)
  //   state_logic = 2;

  // start forward token by token
  for (int input_idx = 0; input_idx < token_length; input_idx++) {
    /**
     * emb forward
     */
    // Initialize the emb input memory
    bm_device_mem_t &emb_in_mem = net_embed->stages[0].input_mems[0];
    bm_device_mem_t &emb_out_mem = net_embed->stages[0].output_mems[0];
    // Input a single token
    bm_memcpy_s2d(bm_handle, emb_in_mem, (void *)&visited_tokens[input_idx]);
    net_launch(net_embed); // forward emb

    /**
     * blocks forward
     */
    // Initialize the blocks memory mapping
    bm_device_mem_t &out0_mem = net_blocks[0]->stages[0].output_mems[0];
    bm_device_mem_t &out1_mem = net_blocks[0]->stages[0].output_mems[1];
    // forward blocks
    for (int idx = 0; idx < NUM_LAYERS; idx++) {
      bm_device_mem_t &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
      bm_device_mem_t &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
      if (idx == 0) // block_0, decide the state handling logic based on the parameters
      {
        if (input_idx ==
            0) // For the first token, initialize the state********************************
        {
          if (!use_previous_state && !use_cached_state) // Use empty state
          {
            bm_memcpy_s2d(bm_handle, in1_mem, (void *)state_init_data.data());
          } else if (use_previous_state && !use_cached_state) // Use in-memory state
          {
            bm_device_mem_t &past_out1_mem =
                net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[1];
            d2d(in1_mem, past_out1_mem);
          } else if (!use_previous_state && use_cached_state) // Use actively cached state
          {
            d2d(in1_mem, state_cache);
          }
        } else // Not the first token, use
               // the in-memory state*************************************************
        {
          bm_device_mem_t &past_out1_mem =
              net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[1];
          d2d(in1_mem, past_out1_mem);
        }
        // First layer; input comes from emb
        d2d(in0_mem, emb_out_mem);
      } else // Not block_0, copy the previous layer's output
      {
        out0_mem = net_blocks[idx - 1]->stages[0].output_mems[0];
        out1_mem = net_blocks[idx - 1]->stages[0].output_mems[1];
        d2d(in0_mem, out0_mem);
        d2d(in1_mem, out1_mem);
      }
      // start forward
      net_launch(net_blocks[idx]);
    }
    // The head is not run during prefill; it only runs at output time
  }
  state_calculated_flag = true; // Having run the blocks once means the state is cached
  if (cache_state2mem) {
    d2d(state_cache, net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[1]);
    state_mem_cached_flag = true; // State has been actively cached
  }
  /**
   * head forward
   */
  // Set up the memory mapping
  bm_device_mem_t &out_mem =
      net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[0];
  bm_device_mem_t &lm_in_mem = net_lm_head->stages[0].input_mems[0];
  bm_device_mem_t &lm_out_mem = net_lm_head->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_head);

  /**
   * sample forward
   */
  uint32_t output_token_temp = 0;
  if (generation_mode == "greedy") {
    output_token_temp = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    output_token_temp = penalty_sample(net_penalty_sample_head, lm_out_mem);
  } else {
    std::cerr << "\nError: Invalid generation mode.\n";
    std::cerr << "Supported modes are 'greedy' or 'penalty_sample'.\n";
    throw std::runtime_error("Invalid generation mode");
  }
  visited_tokens[token_length] = output_token_temp;
  token_length += 1;
  return output_token_temp;
}

/**
 * @brief Recurrent inference: perform single-token inference using the data already in the model memory
 *
 * @param input_token New input token (usually the token just generated)
 * @param use_cached_state  Whether to use the actively cached state
 * @param cache_state2mem   Whether to actively cache the state
 * @return int
 */
int RWKV6::rnn_gen(bool use_cached_state, bool cache_state2mem) {
  int cur_token = visited_tokens[token_length - 1];

  // RNN requires a cached state; using the in-model state without a prior prefill is forbidden
  assert(state_calculated_flag == true);

  /**
   * emb forward
   */
  bm_device_mem_t &emb_in_mem = net_embed->stages[0].input_mems[0];
  bm_device_mem_t &emb_out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, emb_in_mem, (void *)&cur_token);
  net_launch(net_embed); // forward emb
  /**
   * blocks forward
   */
  bm_device_mem_t &out0_mem = net_blocks[0]->stages[0].output_mems[0]; // Input
  bm_device_mem_t &out1_mem = net_blocks[0]->stages[0].output_mems[1];
  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    bm_device_mem_t &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    bm_device_mem_t &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    if (idx == 0) // block_0, copy the state and start generating
    {
      if (use_cached_state) // Use actively cached state
      {
        d2d(in1_mem, state_cache);
      } else // Use in-memory state
      {
        bm_device_mem_t &past_out1_mem =
            net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[1];
        d2d(in1_mem, past_out1_mem); // Initialize state to the state from the previous run
        d2d(in0_mem, emb_out_mem);
      }    // First layer; input comes from emb
    } else // Not the first layer, copy the previous layer's output
    {
      out0_mem = net_blocks[idx - 1]->stages[0].output_mems[0];
      out1_mem = net_blocks[idx - 1]->stages[0].output_mems[1];
      d2d(in0_mem, out0_mem);
      d2d(in1_mem, out1_mem);
    }
    net_launch(net_blocks[idx]); // start forward
  }
  if (cache_state2mem) {
    d2d(state_cache, net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[1]);
    state_mem_cached_flag = true; // State has been actively cached
  }
  /**
   * head forward
   */
  // Set up the memory mapping
  bm_device_mem_t &out_mem =
      net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[0];
  bm_device_mem_t &lm_in_mem = net_lm_head->stages[0].input_mems[0];
  bm_device_mem_t &lm_out_mem = net_lm_head->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_head);

  /**
   * sample forward
   */
  uint32_t output_token_temp = 0;
  if (generation_mode == "greedy") {
    output_token_temp = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    output_token_temp = penalty_sample(net_penalty_sample_head, lm_out_mem);
  } else {
    std::cerr << "\nError: Invalid generation mode.\n";
    std::cerr << "Supported modes are 'greedy' or 'penalty_sample'.\n";
    throw std::runtime_error("Invalid generation mode");
  }
  visited_tokens[token_length] = output_token_temp;
  token_length += 1;
  return output_token_temp;
}

std::vector<int> RWKV6::generate(std::vector<uint32_t> &input_tokens, int EOS) {
  if (input_tokens.empty()) {
    printf("Sorry: your input is empty!!\n");
    input_tokens.clear();
    return {};
  }
  // Limit the maximum number of tokens (though it may not be very useful)
  if ((int)input_tokens.size() > SEQLEN - 10) {
    input_tokens.clear();
    printf("Error: your question is too large!\n");
    return {};
  }
  std::vector<int> result_tokens;
  int token = prefill(input_tokens);
  while (token != EOS && token_length < SEQLEN) {
    result_tokens.emplace_back(token);
    token = rnn_gen(token);
  }
  return result_tokens;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<RWKV6>(m, "RWKV6")
      .def(pybind11::init<>())
      .def("init", &RWKV6::init)
      .def("prefill", &RWKV6::prefill)
      .def("rnn_gen", &RWKV6::rnn_gen)
      .def("generate", &RWKV6::generate)
      .def("deinit", &RWKV6::deinit)
      .def_readwrite("SEQLEN", &RWKV6::SEQLEN) // read SEQLEN in pipeline.py
      .def_readwrite("token_length", &RWKV6::token_length)
      .def_readwrite("temperature", &RWKV6::temperature)
      .def_readwrite("top_p", &RWKV6::top_p)
      .def_readwrite("repeat_penalty", &RWKV6::repeat_penalty)
      .def_readwrite("repeat_last_n", &RWKV6::repeat_last_n)
      .def_readwrite("max_new_tokens", &RWKV6::max_new_tokens)
      .def_readwrite("generation_mode", &RWKV6::generation_mode)
      .def_readwrite("prompt_mode", &RWKV6::prompt_mode);
}