//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// Chip 2: Block / Block_Cache worker for distributed Qwen3_VL pipeline.
// Listens for data from Step0, runs transformer blocks, sends hidden states
// to Step2, and relays tokens back to Step0.
//
//===----------------------------------------------------------------------===//

#include "bmruntime_interface.h"
#include "memory.h"
#include "net_helper.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <vector>

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

//===------------------------------------------------------------===//
// Step1 Block Runner
//===------------------------------------------------------------===//
class Step1Block {
public:
  void init(int devid, const std::string &model_path);
  void deinit();
  void clear_history();

  // Prefill: run all blocks, return last-token hidden state
  std::vector<uint16_t> forward_blocks_prefill(
      int token_length, const std::vector<int> &position_ids_orig,
      const std::vector<uint16_t> &hidden_states, int num_deepstack_recv,
      const std::vector<std::vector<uint16_t>> &deepstacks);

  // Decode: run all block_cache layers, return hidden state
  std::vector<uint16_t>
  forward_blocks_decode(const int position_ids[3],
                        const std::vector<uint16_t> &hidden_state);

  int HIDDEN_SIZE;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int KV_BYTES;
  int NUM_LAYERS;
  bool support_history;
  bool is_dynamic;
  uint16_t mask_value = 0xF0E2;
  int history_length = 0;
  int num_deepstack;
  bm_handle_t bm_handle;

private:
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_add = nullptr;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> deepstack_buffers;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;

  void init_tensors(const bm_net_info_t *net,
                    std::vector<bm_tensor_t> &in_tensors,
                    std::vector<bm_tensor_t> &out_tensors, int stage = 0);
  void net_launch(const bm_net_info_t *net,
                  const std::vector<bm_tensor_t> &in_tensors,
                  std::vector<bm_tensor_t> &out_tensors);
  void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
           int size = 0);

  // Internal prefill paths
  std::vector<uint16_t>
  forward_first_no_kv(int token_length,
                      const std::vector<int> &position_ids_orig,
                      int num_deepstack_recv);
  std::vector<uint16_t>
  forward_first_with_kv(int token_length,
                        const std::vector<int> &position_ids_orig,
                        int num_deepstack_recv);
};

void Step1Block::init_tensors(const bm_net_info_t *net,
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

void Step1Block::net_launch(const bm_net_info_t *net,
                            const std::vector<bm_tensor_t> &in_tensors,
                            std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
}

void Step1Block::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                     int size) {
  if (!size)
    size = bm_mem_get_device_size(src);
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Step1Block::init(int devid, const std::string &model_path) {
  printf("[Step1] Device [%d] loading ...\n", devid);
  bm_status_t status = bm_dev_request(&bm_handle, devid);
  assert(BM_SUCCESS == status);

  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  printf("[Step1] Model [%s] loading ...\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  bm_thread_sync(bm_handle);
  printf("[Step1] Model loaded.\n");

  // Discover networks
  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0)
        return true;
    }
    return false;
  };
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);

  // Find net_add
  if (is_exist("add", net_names, num_nets)) {
    net_add = bmrt_get_network_info(p_bmrt, "add");
  }

  // Count block networks
  int block_count = 0;
  while (true) {
    auto bname = "block_" + std::to_string(block_count);
    auto cname = "block_cache_" + std::to_string(block_count);
    if (!is_exist(bname.c_str(), net_names, num_nets) ||
        !is_exist(cname.c_str(), net_names, num_nets))
      break;
    net_blocks.push_back(bmrt_get_network_info(p_bmrt, bname.c_str()));
    net_blocks_cache.push_back(bmrt_get_network_info(p_bmrt, cname.c_str()));
    block_count++;
  }
  free(net_names);
  NUM_LAYERS = block_count;
  printf("[Step1] Num Layers: %d\n", NUM_LAYERS);
  assert(NUM_LAYERS > 0);

  // Determine model parameters from block shapes
  support_history = net_blocks[0]->input_num == 5;
  is_dynamic = net_blocks[0]->is_dynamic;
  MAX_INPUT_LENGTH = net_blocks[0]->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_blocks[0]->stages[0].input_shapes[0].dims[2];
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  KV_BYTES =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  PREFILL_KV_LENGTH = 0;
  if (support_history) {
    PREFILL_KV_LENGTH = net_blocks[0]->stages[0].input_shapes[3].dims[1];
  }

  // Determine mask_value from hidden-state dtype.
  // Single-device code uses embed_cache output dtype; here we align with that
  // by using the block hidden-state input dtype rather than the attention-mask
  // input dtype.
  {
    auto dtype = net_blocks[0]->input_dtypes[0];
    if (dtype == BM_FLOAT16)
      mask_value = 0xF0E2;
    else if (dtype == BM_BFLOAT16)
      mask_value = 0xC61C;
    else {
      std::cerr << "[Step1] Error: Invalid hidden-state dtype" << std::endl;
      mask_value = 0xF0E2; // default
    }
  }

  printf("[Step1] HIDDEN_SIZE=%d, SEQLEN=%d, MAX_INPUT_LENGTH=%d\n",
         HIDDEN_SIZE, SEQLEN, MAX_INPUT_LENGTH);
  printf("[Step1] support_history=%d, is_dynamic=%d, hidden_dtype=%d, "
         "mask_value=0x%04X\n",
         support_history, is_dynamic, net_blocks[0]->input_dtypes[0],
         mask_value);

  // Allocate KV cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }

  // Allocate device buffer for hidden states
  auto buffer_size = MAX_INPUT_LENGTH * HIDDEN_SIZE * sizeof(uint16_t);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);

  // Determine deepstack support
  num_deepstack = net_add ? NUM_LAYERS : 0; // conservative estimate
  // Actually num_deepstack comes from vit output_num - 1, but step1 doesn't
  // have vit. We'll use what step0 tells us per request.
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_device_mem_t mem;
    status = bm_malloc_device_byte(bm_handle, &mem, buffer_size);
    assert(BM_SUCCESS == status);
    deepstack_buffers.push_back(mem);
  }

  history_length = 0;
}

void Step1Block::deinit() {
  for (auto &mem : deepstack_buffers)
    bm_free_device(bm_handle, mem);
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void Step1Block::clear_history() {
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
}

std::vector<uint16_t>
Step1Block::forward_first_no_kv(int token_length,
                                const std::vector<int> &position_ids_orig,
                                int num_deepstack_recv) {
  // Compute padded position_ids and attention_mask
  std::vector<int> position_ids_pad;
  std::vector<uint16_t> attention_mask;
  int ori_length = position_ids_orig.size() / 3;

  if (is_dynamic) {
    attention_mask.assign(token_length * token_length, mask_value);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * token_length + j] = 0;
      }
    }
    position_ids_pad.assign(3 * token_length, 0);
    std::copy(position_ids_orig.begin(), position_ids_orig.end(),
              position_ids_pad.begin());
  } else {
    int length = MAX_INPUT_LENGTH;
    attention_mask.assign(length * length, mask_value);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * length + j] = 0;
      }
    }
    position_ids_pad.assign(3 * length, 0);
    for (int i = 0; i < 3; i++) {
      int ori_offset = i * ori_length;
      int dst_offset = i * length;
      std::copy(position_ids_orig.data() + ori_offset,
                position_ids_orig.data() + ori_offset + ori_length,
                position_ids_pad.begin() + dst_offset);
    }
  }

  // Run blocks
  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors);
    if (is_dynamic) {
      d2d(in_tensors[0].device_mem, out_mem);
      bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                            (void *)position_ids_pad.data(),
                            token_length * 3 * sizeof(int));
      bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                            (void *)attention_mask.data(),
                            token_length * token_length * sizeof(uint16_t));
      in_tensors[0].shape.dims[1] = token_length;
      in_tensors[1].shape.dims[1] = token_length;
      in_tensors[2].shape.dims[2] = token_length;
      in_tensors[2].shape.dims[3] = token_length;
    } else {
      in_tensors[0].device_mem = out_mem;
      bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                    (void *)position_ids_pad.data());
      bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                    (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx], in_tensors, out_tensors);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];

    // Deepstack add (only for first num_deepstack_recv layers, matching
    // single-device)
    if (net_add && (idx < num_deepstack_recv)) {
      init_tensors(net_add, in_tensors, out_tensors);
      in_tensors[0].device_mem = out_mem;
      in_tensors[1].device_mem = deepstack_buffers[idx];
      net_launch(net_add, in_tensors, out_tensors);
      out_mem = net_add->stages[0].output_mems[0];
    }

    // Save KV cache
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], 0,
                       net_blocks[idx]->stages[0].output_mems[1], 0,
                       KV_BYTES * token_length);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], 0,
                       net_blocks[idx]->stages[0].output_mems[2], 0,
                       KV_BYTES * token_length);
  }

  history_length = token_length;

  // Extract last-token hidden state
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  std::vector<uint16_t> last_hidden(HIDDEN_SIZE);
  bm_device_mem_t last_mem = bm_mem_from_device(
      out_mem.u.device.device_addr + (token_length - 1) * bytes, bytes);
  bm_memcpy_d2s(bm_handle, (void *)last_hidden.data(), last_mem);
  return last_hidden;
}

std::vector<uint16_t>
Step1Block::forward_first_with_kv(int token_length,
                                  const std::vector<int> &position_ids_orig,
                                  int num_deepstack_recv) {
  int max_kv_length = MAX_INPUT_LENGTH + PREFILL_KV_LENGTH;
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * max_kv_length,
                                       mask_value);
  auto old_length = history_length;
  history_length += token_length;
  assert(history_length < SEQLEN);
  assert(old_length <= PREFILL_KV_LENGTH);

  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < old_length; j++) {
      attention_mask[i * max_kv_length + j] = 0;
    }
    for (int j = 0; j <= i; j++) {
      attention_mask[i * max_kv_length + j + PREFILL_KV_LENGTH] = 0;
    }
  }

  int ori_length = position_ids_orig.size() / 3;
  assert(ori_length == token_length);
  assert(ori_length <= MAX_INPUT_LENGTH);
  std::vector<int> position_ids_pad(3 * MAX_INPUT_LENGTH, 0);
  for (int i = 0; i < 3; i++) {
    int ori_offset = i * ori_length;
    int dst_offset = i * MAX_INPUT_LENGTH;
    std::copy(position_ids_orig.data() + ori_offset,
              position_ids_orig.data() + ori_offset + ori_length,
              position_ids_pad.begin() + dst_offset);
  }

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;

  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors);
    in_tensors[0].device_mem = out_mem;
    if (old_length > 0) {
      d2d(in_tensors[3].device_mem, past_key[idx], 0, KV_BYTES * old_length);
      d2d(in_tensors[4].device_mem, past_value[idx], 0, KV_BYTES * old_length);
    } else if (idx == 0) {
      empty(bm_handle, in_tensors[3].device_mem);
      empty(bm_handle, in_tensors[4].device_mem);
    }
    bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                  (void *)position_ids_pad.data());
    bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                  (void *)attention_mask.data());
    net_launch(net_blocks[idx], in_tensors, out_tensors);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];

    // Deepstack add (only for first num_deepstack_recv layers, matching
    // single-device)
    if (net_add && (idx < num_deepstack_recv)) {
      init_tensors(net_add, in_tensors, out_tensors);
      in_tensors[0].device_mem = out_mem;
      in_tensors[1].device_mem = deepstack_buffers[idx];
      net_launch(net_add, in_tensors, out_tensors);
      out_mem = net_add->stages[0].output_mems[0];
    }

    auto &out1_mem = net_blocks[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks[idx]->stages[0].output_mems[2];
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], old_length * KV_BYTES,
                       out1_mem, 0, KV_BYTES * token_length);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], old_length * KV_BYTES,
                       out2_mem, 0, KV_BYTES * token_length);
  }

  // Extract last-token hidden state
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  std::vector<uint16_t> last_hidden(HIDDEN_SIZE);
  bm_device_mem_t last_mem = bm_mem_from_device(
      out_mem.u.device.device_addr + (token_length - 1) * bytes, bytes);
  bm_memcpy_d2s(bm_handle, (void *)last_hidden.data(), last_mem);
  return last_hidden;
}

std::vector<uint16_t> Step1Block::forward_blocks_prefill(
    int token_length, const std::vector<int> &position_ids_orig,
    const std::vector<uint16_t> &hidden_states, int num_deepstack_recv,
    const std::vector<std::vector<uint16_t>> &deepstacks) {

  // Upload hidden_states to device
  int hs_bytes = token_length * HIDDEN_SIZE * sizeof(uint16_t);
  empty(bm_handle, dev_buffer);
  bm_memcpy_s2d_partial(bm_handle, dev_buffer, (void *)hidden_states.data(),
                        hs_bytes);

  // Upload deepstack buffers to device
  for (int i = 0; i < num_deepstack_recv && i < (int)deepstack_buffers.size();
       i++) {
    empty(bm_handle, deepstack_buffers[i]);
    bm_memcpy_s2d_partial(bm_handle, deepstack_buffers[i],
                          (void *)deepstacks[i].data(), hs_bytes);
  }

  // Choose prefill path
  if (support_history) {
    return forward_first_with_kv(token_length, position_ids_orig,
                                 num_deepstack_recv);
  } else {
    return forward_first_no_kv(token_length, position_ids_orig,
                               num_deepstack_recv);
  }
}

std::vector<uint16_t>
Step1Block::forward_blocks_decode(const int position_ids[3],
                                  const std::vector<uint16_t> &hidden_state) {
  // Compute attention mask for decode
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = history_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }

  // Upload hidden_state to dev_buffer (separate from any block output mem)
  bm_memcpy_s2d_partial(bm_handle, dev_buffer, (void *)hidden_state.data(),
                        HIDDEN_SIZE * sizeof(uint16_t));

  // KV offset
  int kv_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (history_length - 1) * kv_bytes;

  auto out_mem = dev_buffer;

  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &net = net_blocks_cache[idx];
    std::vector<bm_tensor_t> in_tensors;
    std::vector<bm_tensor_t> out_tensors;
    init_tensors(net, in_tensors, out_tensors);

    in_tensors[0].device_mem = out_mem;
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem, (void *)position_ids);
      bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                    (void *)attention_mask.data());
    } else {
      in_tensors[1].device_mem = net_blocks_cache[0]->stages[0].input_mems[1];
      in_tensors[2].device_mem = net_blocks_cache[0]->stages[0].input_mems[2];
    }
    out_tensors[1].device_mem = bm_mem_from_device(
        past_key[idx].u.device.device_addr + token_offset, KV_BYTES);
    out_tensors[2].device_mem = bm_mem_from_device(
        past_value[idx].u.device.device_addr + token_offset, KV_BYTES);

    net_launch(net, in_tensors, out_tensors);
    out_mem = net->stages[0].output_mems[0];
  }

  history_length++;

  // Copy output hidden state to host
  std::vector<uint16_t> out_hidden(HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, (void *)out_hidden.data(), out_mem,
                        HIDDEN_SIZE * sizeof(uint16_t));
  return out_hidden;
}

//===------------------------------------------------------------===//
// Main event loop
//===------------------------------------------------------------===//
void Usage() {
  printf("Step1 (Block Worker) Usage:\n"
         "  -m, --model     : Block bmodel path\n"
         "  -d, --devid     : Device ID (default 0)\n"
         "  -p, --port      : Listen port for Step0 (default 10001)\n"
         "  --step2_host    : Step2 host (default 127.0.0.1)\n"
         "  --step2_port    : Step2 port (default 10002)\n"
         "  -h, --help      : Show help\n");
}

int main(int argc, char *argv[]) {
  std::string model_path;
  int devid = 0;
  int listen_port = 10001;
  std::string step2_host = "127.0.0.1";
  int step2_port = 10002;

  struct option long_opts[] = {{"model", required_argument, nullptr, 'm'},
                               {"devid", required_argument, nullptr, 'd'},
                               {"port", required_argument, nullptr, 'p'},
                               {"step2_host", required_argument, nullptr, 1},
                               {"step2_port", required_argument, nullptr, 2},
                               {"help", no_argument, nullptr, 'h'},
                               {nullptr, 0, nullptr, 0}};
  int opt;
  while ((opt = getopt_long(argc, argv, "m:d:p:h", long_opts, nullptr)) != -1) {
    switch (opt) {
    case 'm':
      model_path = optarg;
      break;
    case 'd':
      devid = atoi(optarg);
      break;
    case 'p':
      listen_port = atoi(optarg);
      break;
    case 1:
      step2_host = optarg;
      break;
    case 2:
      step2_port = atoi(optarg);
      break;
    case 'h':
      Usage();
      return 0;
    default:
      Usage();
      return 1;
    }
  }
  if (model_path.empty()) {
    Usage();
    return 1;
  }

  // Initialize model
  Step1Block block;
  block.init(devid, model_path);

  // Start listening for Step0
  int server_fd = create_server(listen_port);
  assert(server_fd >= 0);
  printf("[Step1] Waiting for Step0 connection on port %d ...\n", listen_port);
  int step0_fd = accept_client(server_fd);
  assert(step0_fd >= 0);
  printf("[Step1] Step0 connected.\n");

  // Connect to Step2
  printf("[Step1] Connecting to Step2 at %s:%d ...\n", step2_host.c_str(),
         step2_port);
  int step2_fd = connect_to(step2_host, step2_port);
  printf("[Step1] Connected to Step2.\n");

  // Send handshake info to Step0
  struct {
    int32_t hidden_size;
    int32_t seqlen;
    int32_t max_input_length;
    int32_t support_history;
    int32_t prefill_kv_length;
  } handshake;
  handshake.hidden_size = block.HIDDEN_SIZE;
  handshake.seqlen = block.SEQLEN;
  handshake.max_input_length = block.MAX_INPUT_LENGTH;
  handshake.support_history = block.support_history ? 1 : 0;
  handshake.prefill_kv_length = block.PREFILL_KV_LENGTH;
  send_all(step0_fd, &handshake, sizeof(handshake));
  printf("[Step1] Handshake sent to Step0.\n");

  // Main loop
  while (true) {
    int32_t msg_type = 0;
    if (!recv_all(step0_fd, &msg_type, sizeof(msg_type))) {
      printf("[Step1] Step0 disconnected.\n");
      break;
    }

    if (msg_type == MSG_SHUTDOWN) {
      printf("[Step1] Shutdown received.\n");
      // Forward shutdown to Step2
      SimpleMsg smsg;
      smsg.msg_type = MSG_SHUTDOWN;
      send_all(step2_fd, &smsg, sizeof(smsg));
      break;
    }

    if (msg_type == MSG_CLEAR) {
      printf("[Step1] Clear history.\n");
      block.clear_history();
      // Forward to Step2
      SimpleMsg smsg;
      smsg.msg_type = MSG_CLEAR;
      send_all(step2_fd, &smsg, sizeof(smsg));
      // Send ack back to Step0
      TokenWithHistory tw;
      tw.token = 0;
      tw.history_length = 0;
      send_all(step0_fd, &tw, sizeof(tw));
      continue;
    }

    if (msg_type == MSG_PREFILL) {
      // Read prefill metadata (remaining fields after msg_type)
      PrefillMeta meta;
      meta.msg_type = msg_type;
      recv_all(step0_fd, &meta.token_length,
               sizeof(PrefillMeta) - sizeof(int32_t));

      // Read position_ids
      std::vector<int> position_ids(meta.position_ids_count);
      recv_all(step0_fd, position_ids.data(),
               meta.position_ids_count * sizeof(int));

      // Read visited_tokens (for sampling)
      std::vector<int> visited_tokens(meta.visited_token_count);
      if (meta.visited_token_count > 0) {
        recv_all(step0_fd, visited_tokens.data(),
                 meta.visited_token_count * sizeof(int));
      }

      // Read hidden_states
      int hs_count = meta.token_length * meta.hidden_size;
      std::vector<uint16_t> hidden_states(hs_count);
      recv_all(step0_fd, hidden_states.data(), hs_count * sizeof(uint16_t));

      // Read deepstack buffers
      int num_ds = meta.num_deepstack;
      std::vector<std::vector<uint16_t>> deepstacks(num_ds);
      for (int i = 0; i < num_ds; i++) {
        deepstacks[i].resize(hs_count);
        recv_all(step0_fd, deepstacks[i].data(), hs_count * sizeof(uint16_t));
      }

      printf("[Step1] Prefill: token_len=%d, deepstack=%d\n", meta.token_length,
             num_ds);

      assert(meta.hidden_size == block.HIDDEN_SIZE);

      // Run blocks
      auto last_hidden = block.forward_blocks_prefill(
          meta.token_length, position_ids, hidden_states, num_ds, deepstacks);
      bm_thread_sync(block.bm_handle);
      // Send to Step2
      LMHeadMeta lm_meta;
      lm_meta.msg_type = MSG_PREFILL;
      lm_meta.hidden_size = meta.hidden_size;
      lm_meta.visited_token_count = meta.visited_token_count;
      send_all(step2_fd, &lm_meta, sizeof(lm_meta));
      if (meta.visited_token_count > 0) {
        send_all(step2_fd, visited_tokens.data(),
                 meta.visited_token_count * sizeof(int));
      }
      send_all(step2_fd, last_hidden.data(),
               meta.hidden_size * sizeof(uint16_t));

      // Receive token from Step2
      TokenMsg tmsg;
      recv_all(step2_fd, &tmsg, sizeof(tmsg));

      // Account for the generated token in history_length
      // Single-device does token_length++ then history_length = token_length
      // after lm_head. Since lm_head runs on Step2, we must do the equivalent
      // increment here so that decode's attention mask and KV offset are
      // correct.
      block.history_length++;

      // Send token + history_length back to Step0
      TokenWithHistory tw;
      tw.token = tmsg.token;
      tw.history_length = block.history_length;
      send_all(step0_fd, &tw, sizeof(tw));

      printf("[Step1] Prefill done, token=%d, history=%d\n", tw.token,
             tw.history_length);
      continue;
    }

    if (msg_type == MSG_DECODE) {
      // Read decode metadata (remaining fields after msg_type)
      DecodeMeta meta;
      meta.msg_type = msg_type;
      recv_all(step0_fd, &meta.position_ids,
               sizeof(DecodeMeta) - sizeof(int32_t));

      // Read hidden_state
      std::vector<uint16_t> hidden_state(meta.hidden_size);
      recv_all(step0_fd, hidden_state.data(),
               meta.hidden_size * sizeof(uint16_t));

      // Run block_cache
      assert(meta.hidden_size == block.HIDDEN_SIZE);
      auto out_hidden =
          block.forward_blocks_decode(meta.position_ids, hidden_state);
      bm_thread_sync(block.bm_handle);
      // Send to Step2
      LMHeadMeta lm_meta;
      lm_meta.msg_type = MSG_DECODE;
      lm_meta.hidden_size = meta.hidden_size;
      lm_meta.visited_token_count = 0;
      send_all(step2_fd, &lm_meta, sizeof(lm_meta));
      send_all(step2_fd, out_hidden.data(),
               meta.hidden_size * sizeof(uint16_t));

      // Receive token from Step2
      TokenMsg tmsg;
      recv_all(step2_fd, &tmsg, sizeof(tmsg));

      // Send back to Step0
      TokenWithHistory tw;
      tw.token = tmsg.token;
      tw.history_length = block.history_length;
      send_all(step0_fd, &tw, sizeof(tw));
      continue;
    }

    printf("[Step1] Unknown message type: %d\n", msg_type);
  }

  block.deinit();
  close(step0_fd);
  close(step2_fd);
  close(server_fd);
  printf("[Step1] Exited.\n");
  return 0;
}
