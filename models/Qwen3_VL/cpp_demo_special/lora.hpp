//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once
#include "bmlib_runtime.h"
#include "safetensors.hh"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

typedef std::pair<bm_device_mem_t, std::shared_ptr<std::vector<uint8_t>>>
    lora_item_t;
typedef std::vector<lora_item_t> lora_cache_t;
typedef std::shared_ptr<lora_cache_t> lora_cache_ptr_t;

class LoraContext {
public:
  LoraContext(const std::string &lora_path, safetensors::dtype lora_type);

  static bool is_lora_path(const std::string &path);

  bool create_lora_item(lora_item_t &lora_item, const std::string &path,
                        bm_handle_t bm_handle, bm_device_mem_t devmem,
                        bool is_embed = false);
  
  bool is_exist(const std::string &path);

  void check_all_tensors_visited() {
    for (int i = 0; i < m_num_tensors; i++) {
      if (!m_tensors_visited[i]) {
        auto key = m_st.tensors.keys()[i];
        std::cout << "Warning: LoRA tensor not used: " << key << std::endl;
      }
    }
  }

protected:
  enum ReadType {
    DO_NOTHING = 0,
    DO_TRANSPOSE_COPY = 1,
    DO_STRIDE_COPY = 2,
    DO_TRANSPOSE_STRIDE_COPY = 3
  };
  struct LoraPath {
    std::string key;
    int rank;
    int dim;
    bool is_A; // A or B
  };
  bool parse_lora_path(const std::string &path, LoraPath &lora_path);

  int get_tensor_index(const std::string &path);

  void read_tensor_data(int tensor_idx, void *dst, size_t size,
                        int max_lora_rank, bool do_scale, ReadType read_type);

  void transpose_copy(uint16_t *dst, uint16_t *src, int rows, int cols,
                      bool do_scale = false);

  void stride_copy(uint16_t *dst, uint16_t *src, int dim0, int dim1,
                   int max_dim1, bool do_scale = false);

  void data_copy(uint16_t *dst, uint16_t *src, size_t num_elems,
                 bool do_scale = false);

  void transpose_stride_copy(uint16_t *dst, uint16_t *src, int dim0, int dim1,
                             int max_dim1, bool do_scale = false);

  uint16_t a16_scale(uint16_t data);

private:
  safetensors::safetensors_t m_st;
  int m_num_tensors;
  float m_scale;
  safetensors::dtype m_lora_type;
  std::vector<bool> m_tensors_visited;
};