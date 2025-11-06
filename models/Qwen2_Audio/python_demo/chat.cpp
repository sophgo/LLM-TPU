//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

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
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <vector>

namespace py = pybind11;
using ArrayFloat =
    py::array_t<float, py::array::c_style | py::array::forcecast>;
using ArrayInt = py::array_t<int, py::array::c_style | py::array::forcecast>;

//===------------------------------------------------------------===//
// Empty Func
//===------------------------------------------------------------===//
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

class Qwen2Audio {
public:
  void init(int devid, std::string model_path);
  void deinit();
  ArrayFloat forward_embed(ArrayInt const &tokens);
  ArrayFloat forward_embed_cache(ArrayInt const &input_ids);
  ArrayFloat forward_audio(ArrayFloat const &input_features,
                   ArrayFloat const &full_attn_mask);
  ArrayFloat forward_head(ArrayFloat const &input_features);
  ArrayFloat forward_project(ArrayFloat const &audio_features);
  std::tuple<std::vector<float>, std::vector<float>, std::vector<float> > 
        forward( ArrayFloat const &input_embeds, ArrayInt const &position_ids, ArrayFloat const &input_attention_mask, const int idx);
  int forward_next(ArrayInt const &position_ids, ArrayInt const &token, ArrayFloat const & attention_mask);
   std::tuple<std::vector<float>, std::vector<float>, std::vector<float> > forward_cache_next(ArrayFloat const &inputs_embeds, 
                            ArrayFloat const &position_ids, 
                            ArrayFloat const & attention_mask,
                            ArrayFloat const &past_key_array,
                            ArrayFloat const &past_value_array,
                            const int idx); ;

  std::mt19937 sgen;
  Qwen2Audio() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  void init_by_names();

public:
  int token_length;
  int SEQLEN; // read from bmodel
  int HIDDEN_SIZE;
  int NUM_LAYERS; // read from bmodel
  int AUDIO_DIMS;
  int max_pos;
  const int spatial_merge_size = 2;
  bool lmhead_with_topk;
  uint16_t mask_value;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_sample_head;
  const bm_net_info_t *net_audio;
  const bm_net_info_t *net_project;
  const bm_net_info_t *net_block_0;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void Qwen2Audio::net_launch(const bm_net_info_t *net, int stage_idx) {
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
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);

  //std::cout << "launch net: " << out_tensors.data() << " ret: " << ret
  //          << std::endl;
  assert(ret);
  bm_thread_sync(bm_handle);
}

void Qwen2Audio::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Qwen2Audio::init_by_names() {
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
  net_audio = bmrt_get_network_info(p_bmrt, "audio_ext_model");
  net_project = bmrt_get_network_info(p_bmrt, "projector");
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greed");
  //net_block_0 = bmrt_get_network_info(p_kv_bmrt, "block_0");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);

  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;

  NUM_LAYERS = 32; // 2 nets for each block, one for cache
  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    if ((!is_exist(block_name.c_str(), net_names, num_nets)) ||
        (!is_exist(cache_name.c_str(), net_names, num_nets))) {
      NUM_LAYERS = i;
      printf("Warning: Only %d blocks found, expected %d blocks.\n", NUM_LAYERS,
             32);
      break;
    }

    //std::cout << "block name: " << block_name << " " << cache_name << std::endl;
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
    //std::cout << "end" << std::endl;
  }
  free(net_names);
}

void Qwen2Audio::init(int dev_id, std::string model_path) {

  // request bm_handle
  std::cout << "Device [ " << dev_id << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  assert(BM_SUCCESS == status);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);

  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");


  init_by_names();
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[2];
  AUDIO_DIMS = net_audio->stages[0].input_shapes[0].dims[1]; // [1, 128, 3000],[1,1,1500, 1500]

  printf("Num Layers:%d\n", NUM_LAYERS);
  printf("HIDDEN_SIZE:%d\n", HIDDEN_SIZE);
  printf("AUDIO_DIMS:%d\n", AUDIO_DIMS);

  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks[i]->stages[0].output_mems[1];
    past_value[i] = net_blocks[i]->stages[0].output_mems[2];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);
  //bmrt_trace(p_kv_bmrt);
}

void Qwen2Audio::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

ArrayFloat Qwen2Audio::forward_embed(ArrayInt const &input_ids) {
  empty_net(bm_handle, net_embed);
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)input_ids.data());
  net_launch(net_embed);
  d2d(dev_buffer, out_mem);
  //std::cout << "embed done\n";


  const size_t total_elements = net_embed->stages[0].output_shapes[0].dims[0] *
                                net_embed->stages[0].output_shapes[0].dims[1] *
                                net_embed->stages[0].output_shapes[0].dims[2] ;

    // 3. 创建一个单层的、连续的 vector 来作为接收缓冲区
  std::vector<float> host_output_buffer(total_elements);
  bm_memcpy_d2s(bm_handle, (void *)host_output_buffer.data(), out_mem);
  return  py::array_t<float>(host_output_buffer.size(), host_output_buffer.data());
  
}

ArrayFloat Qwen2Audio::forward_embed_cache(ArrayInt const &input_ids) {
  empty_net(bm_handle, net_embed_cache);
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)input_ids.data());
  net_launch(net_embed_cache);

    // 2. 计算总共需要多少个 float 元素
  const size_t total_elements = net_embed_cache->stages[0].output_shapes[0].dims[0] *
                                net_embed_cache->stages[0].output_shapes[0].dims[1] *
                                net_embed_cache->stages[0].output_shapes[0].dims[2] ;

    // 3. 创建一个单层的、连续的 vector 来作为接收缓冲区
  std::vector<float> host_output_buffer(total_elements);

  bm_memcpy_d2s(bm_handle, (void *)host_output_buffer.data(), out_mem);
  return  py::array_t<float>(host_output_buffer.size(), host_output_buffer.data());
  
}

ArrayFloat Qwen2Audio::forward_audio(ArrayFloat const &input_features, ArrayFloat const &attention_maks) {
  empty_net(bm_handle, net_audio);
  auto &audio_in0_mem = net_audio->stages[0].input_mems[0];
  auto &audio_in1_mem = net_audio->stages[0].input_mems[1];
  auto &audio_out_mem = net_audio->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, audio_in0_mem, (void *)input_features.data());
  bm_memcpy_s2d(bm_handle, audio_in1_mem, (void *)attention_maks.data());
  net_launch(net_audio);
  d2d(dev_buffer, audio_out_mem);

    // 2. 计算总共需要多少个 float 元素
  const size_t total_elements = net_audio->stages[0].output_shapes[0].dims[0] *
                                net_audio->stages[0].output_shapes[0].dims[1] *
                                net_audio->stages[0].output_shapes[0].dims[2] ;

  std::vector<float> host_output_buffer(total_elements);

  bm_memcpy_d2s(bm_handle, (void *)host_output_buffer.data(), audio_out_mem);
  return  py::array_t<float>(host_output_buffer.size(), host_output_buffer.data());
}

ArrayFloat Qwen2Audio::forward_project(ArrayFloat const &audio_features) {
  empty_net(bm_handle, net_project);
  auto &project_in_mem = net_project->stages[0].input_mems[0];
  auto &project_out_mem = net_project->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, project_in_mem, (void *)audio_features.data());
  net_launch(net_project);
  d2d(dev_buffer, project_out_mem);

    // 2. 计算总共需要多少个 float 元素
  const size_t total_elements = net_project->stages[0].output_shapes[0].dims[0] *
                                net_project->stages[0].output_shapes[0].dims[1] *
                                net_project->stages[0].output_shapes[0].dims[2] ;

  std::vector<float> host_output_buffer(total_elements);

  bm_memcpy_d2s(bm_handle, (void *)host_output_buffer.data(), project_out_mem);
  return  py::array_t<float>(host_output_buffer.size(), host_output_buffer.data());
}
void Qwen2Audio::head_launch(const bm_net_info_t *net,
                          bm_device_mem_t &logits_mem) {
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
  bm_thread_sync(bm_handle);
}

int Qwen2Audio::greedy_search(const bm_net_info_t *net,
                           bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

ArrayFloat Qwen2Audio::forward_head(ArrayFloat const &input_features) {
  empty_net(bm_handle, net_lm);

  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  //std::cout << "forward lmhead\n";
  bm_memcpy_s2d(bm_handle, lm_in_mem, (void *)input_features.data());
  net_launch(net_lm);
  //std::cout << "launch end\n";
  const size_t lm_total_elements = net_lm->stages[0].output_shapes[0].dims[0] *
                                net_lm->stages[0].output_shapes[0].dims[1] *
                                net_lm->stages[0].output_shapes[0].dims[2] ;
  std::vector<float> lm_output_buffer(lm_total_elements);
  bm_memcpy_d2s(bm_handle, (void *)lm_output_buffer.data(), lm_out_mem);
  return  py::array_t<float>(lm_output_buffer.size(), lm_output_buffer.data());
}
 std::tuple<std::vector<float>, std::vector<float>, std::vector<float> > 
    Qwen2Audio::forward( ArrayFloat const &input_embeds, ArrayInt const &position_ids, ArrayFloat const &input_attention_mask,const int idx) {
  empty_net(bm_handle, net_blocks[idx]);
  
  auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
  auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
  auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];

  auto & out0_mem = net_blocks[idx]->stages[0].output_mems[0];
  auto & out1_mem = net_blocks[idx]->stages[0].output_mems[1];
  auto & out2_mem = net_blocks[idx]->stages[0].output_mems[2];

  bm_memcpy_s2d(bm_handle, in0_mem, (void *)input_embeds.data());
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_ids.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)input_attention_mask.data());

  net_launch(net_blocks[idx]);

  const size_t kv_total_elements = net_blocks[idx]->stages[0].output_shapes[2].dims[0] *
                                    net_blocks[idx]->stages[0].output_shapes[2].dims[1] *
                                    net_blocks[idx]->stages[0].output_shapes[2].dims[2] *
                                    net_blocks[idx]->stages[0].output_shapes[2].dims[3];
  std::vector<float> v_host_output_buffer(kv_total_elements);
  bm_memcpy_d2s(bm_handle, (void *)v_host_output_buffer.data(), out2_mem);

  std::vector<float> k_host_output_buffer(kv_total_elements);
  bm_memcpy_d2s(bm_handle, (void *)k_host_output_buffer.data(), out1_mem);

  const size_t inputs_embeds_total_elements = net_blocks[idx]->stages[0].output_shapes[0].dims[0] *
                                    net_blocks[idx]->stages[0].output_shapes[0].dims[1] *
                                    net_blocks[idx]->stages[0].output_shapes[0].dims[2];
  //std::cout << "\n " << inputs_embeds_total_elements << " " << std::endl;
  std::vector<float> input_embeds_host_input_buffer(inputs_embeds_total_elements);
  bm_memcpy_d2s(bm_handle, (void *)input_embeds_host_input_buffer.data(), out0_mem);

  return std::make_tuple(input_embeds_host_input_buffer, k_host_output_buffer, v_host_output_buffer);
 }

 std::tuple<std::vector<float>, std::vector<float>, std::vector<float> > Qwen2Audio::forward_cache_next(ArrayFloat const &inputs_embeds, 
                            ArrayFloat const &position_ids, 
                            ArrayFloat const & attention_mask,
                            ArrayFloat const & past_key_array,
                            ArrayFloat const & past_value_array,
                            const int idx) {
  //empty_net(bm_handle, net_blocks_cache[idx]);

  auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
  auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
  auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
  auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
  auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];

  auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
  auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
  auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];

  bm_memcpy_s2d(bm_handle, in0_mem, (void *)inputs_embeds.data());
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_ids.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
  bm_memcpy_s2d(bm_handle, in3_mem, (void *)past_key_array.data());
  bm_memcpy_s2d(bm_handle, in4_mem, (void *)past_value_array.data());

  net_launch(net_blocks_cache[idx]);

  const size_t v_total_elements = net_blocks_cache[idx]->stages[0].output_shapes[2].dims[0] *
                                    net_blocks_cache[idx]->stages[0].output_shapes[2].dims[1] *
                                    net_blocks_cache[idx]->stages[0].output_shapes[2].dims[2] *
                                    net_blocks_cache[idx]->stages[0].output_shapes[2].dims[3];


  std::vector<float> v_host_output_buffer(v_total_elements);
  bm_memcpy_d2s(bm_handle, (void *)v_host_output_buffer.data(), out2_mem);

  std::vector<float> k_host_output_buffer(v_total_elements);
  bm_memcpy_d2s(bm_handle, (void *)k_host_output_buffer.data(), out1_mem);

  const size_t embedding_total_elements = net_blocks_cache[idx]->stages[0].output_shapes[0].dims[0] *
                                    net_blocks_cache[idx]->stages[0].output_shapes[0].dims[1] *
                                    net_blocks_cache[idx]->stages[0].output_shapes[0].dims[2];
  std::vector<float> input_embeds_host_input_buffer(embedding_total_elements);
  bm_memcpy_d2s(bm_handle, (void *)input_embeds_host_input_buffer.data(), out0_mem);
  //return py::array_t<float>(input_embeds_host_input_buffer.size(), input_embeds_host_input_buffer.data(), py::cast(new std::vector<float>(std::move(input_embeds_host_input_buffer))));
  return std::make_tuple(input_embeds_host_input_buffer, k_host_output_buffer, v_host_output_buffer);

}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen2Audio>(m, "Qwen2Audio")
      .def(pybind11::init<>())
      .def("init", &Qwen2Audio::init)
      .def("forward_embed", &Qwen2Audio::forward_embed)
      .def("forward_embed_cache", &Qwen2Audio::forward_embed_cache)
      .def("forward_audio", &Qwen2Audio::forward_audio)
      .def("forward_project", &Qwen2Audio::forward_project)
      .def("forward_head", &Qwen2Audio::forward_head)
      .def("forward_cache_next", &Qwen2Audio::forward_cache_next)
      .def("forward", &Qwen2Audio::forward)
      .def("deinit", &Qwen2Audio::deinit)
      .def_readonly("SEQLEN", &Qwen2Audio::SEQLEN) // read SEQLEN in pipeline.py
      .def_readonly("NUM_LAYERS", &Qwen2Audio::NUM_LAYERS)
      .def_readwrite("token_length", &Qwen2Audio::token_length);
}
