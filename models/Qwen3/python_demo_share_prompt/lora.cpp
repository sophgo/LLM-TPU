//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "lora.hpp"
#include "json.hpp"
#include <fstream>

LoraContext::LoraContext(const std::string &lora_path,
                         safetensors::dtype lora_type) {
  std::ifstream in(lora_path + "/adapter_config.json");
  nlohmann::json j;
  in >> j;
  int lora_alpha = 1;
  if (j.contains("lora_alpha")) {
    lora_alpha = j["lora_alpha"].get<int>();
  }
  int r = 32;
  if (j.contains("r")) {
    r = j["r"].get<int>();
  }
  m_scale = static_cast<float>(lora_alpha) / static_cast<float>(r);

  std::string warn, err;
  bool ret = safetensors::load_from_file(
      lora_path + "/adapter_model.safetensors", &m_st, &warn, &err);
  if (warn.empty() == false) {
    std::cout << "Warning: " << warn << std::endl;
  }
  if (ret == false) {
    std::cerr << "Error: Load LoRA safetensors failed: " << err << std::endl;
    throw std::runtime_error("Load LoRA safetensors failed");
  }
  // Check if data_offsets are valid.
  if (!safetensors::validate_data_offsets(m_st, err)) {
    std::cerr << "Invalid data_offsets: " << err << "\n";
    throw std::runtime_error("Invalid data_offsets");
  }
  m_num_tensors = m_st.tensors.size();
  m_tensors_visited.assign(m_num_tensors, false);
  m_lora_type = lora_type;
}

bool LoraContext::is_lora_path(const std::string &path) {
  if (path.rfind("lora.", 0) != 0) {
    return false;
  }
  return true;
}

// true: load data to device; false: not find lora tensor
bool LoraContext::load_lora_to_device(const std::string &path,
                                      bm_handle_t bm_handle,
                                      bm_device_mem_t devmem, bool is_embed) {
  LoraPath loraPath;
  if (!parse_lora_path(path, loraPath)) {
    return false;
  }
  int tensor_idx = get_tensor_index(loraPath.key);
  if (tensor_idx < 0) {
    int value = 0;
    auto ret = bm_memset_device_ext(bm_handle, &value, 1, devmem);
    if (ret != BM_SUCCESS) {
      throw std::runtime_error("bm_memset_device_ext failed");
    }
    return false;
  }
  size_t dev_size = bm_mem_get_device_size(devmem);
  uint8_t *buffer = new uint8_t[dev_size];
  if (buffer == nullptr) {
    throw std::runtime_error("malloc failed");
  }
  ReadType read_type = DO_NOTHING;
  if (loraPath.is_A == false) {
    // lora_B
    if (loraPath.dim == 0) {
      read_type = DO_TRANSPOSE_COPY;
    } else {
      read_type = DO_STRIDE_COPY;
    }
  } else {
    // lora_A
    if (is_embed) {
      read_type = DO_STRIDE_COPY;
    } else if (loraPath.dim == 1) {
      read_type = DO_TRANSPOSE_STRIDE_COPY;
    } else {
      read_type = DO_NOTHING;
    }
  }
  bool do_scale = false;
  if (loraPath.is_A == false && m_scale != 1.0f) {
    do_scale = true;
  }
  read_tensor_data(tensor_idx, buffer, dev_size, loraPath.rank, do_scale,
                   read_type);
  auto ret = bm_memcpy_s2d(bm_handle, devmem, buffer);
  if (ret != BM_SUCCESS) {
    throw std::runtime_error("bm_memset_device_ext failed");
  }
  delete[] buffer;
  return true;
}

bool LoraContext::parse_lora_path(const std::string &path,
                                  LoraPath &lora_path) {
  if (!is_lora_path(path)) {
    return false;
  }
  size_t first_dot = path.find('.', 5);
  auto rank_str = path.substr(5, first_dot - 5);
  lora_path.rank = std::stoi(rank_str);
  size_t second_dot = path.find('.', first_dot + 1);
  auto dim_str = path.substr(first_dot + 1, second_dot - first_dot - 1);
  lora_path.dim = std::stoi(dim_str);
  lora_path.key = path.substr(second_dot + 1);
  lora_path.is_A = (lora_path.key.find(".lora_A") != std::string::npos);
  return true;
}

int LoraContext::get_tensor_index(const std::string &path) {
  int idx = 0;
  for (; idx < m_num_tensors; idx++) {
    auto key = m_st.tensors.keys()[idx];
    if (true == std::equal(path.rbegin(), path.rend(), key.rbegin())) {
      break;
    }
  }
  if (idx < m_num_tensors) {
    return idx;
  }
  return -1;
}

void LoraContext::read_tensor_data(int tensor_idx, void *dst, size_t size,
                                   int max_lora_rank, bool do_scale,
                                   ReadType read_type) {
  safetensors::tensor_t tensor;
  m_st.tensors.at(tensor_idx, &tensor);
  if (tensor.dtype != m_lora_type) {
    throw std::runtime_error("LoRA tensor dtype mismatch");
  }
  auto bytes = tensor.data_offsets[1] - tensor.data_offsets[0];
  if (bytes > size) {
    throw std::runtime_error("Buffer size is smaller than tensor data size");
  }
  const uint8_t *data_ptr = m_st.storage.data() + tensor.data_offsets[0];
  std::memset(dst, 0, size);
  switch (read_type) {
  case DO_TRANSPOSE_COPY:
    transpose_copy((uint16_t *)dst, (uint16_t *)data_ptr, tensor.shape[0],
                   tensor.shape[1], do_scale);
    break;
  case DO_STRIDE_COPY:
    if (tensor.shape[1] > static_cast<size_t>(max_lora_rank)) {
      throw std::runtime_error(
          "LoRA tensor shape[1] is larger than max_lora_rank");
    }
    stride_copy((uint16_t *)dst, (uint16_t *)data_ptr, tensor.shape[0],
                tensor.shape[1], max_lora_rank, do_scale);
    break;
  case DO_TRANSPOSE_STRIDE_COPY:
    if (tensor.shape[0] > static_cast<size_t>(max_lora_rank)) {
      throw std::runtime_error(
          "LoRA tensor shape[0] is larger than max_lora_rank");
    }
    transpose_stride_copy((uint16_t *)dst, (uint16_t *)data_ptr,
                          tensor.shape[0], tensor.shape[1], max_lora_rank,
                          do_scale);
    break;
  default:
    data_copy((uint16_t *)dst, (uint16_t *)data_ptr, bytes / 2, do_scale);
    break;
  }
  m_tensors_visited[tensor_idx] = true;
}

uint16_t LoraContext::a16_scale(uint16_t data) {
  if (m_lora_type == safetensors::dtype::kFLOAT16) {
    float val = safetensors::fp16_to_float(data);
    val *= m_scale;
    return safetensors::float_to_fp16(val);
  } else if (m_lora_type == safetensors::dtype::kBFLOAT16) {
    // Convert BF16 to float
    float val = safetensors::bfloat16_to_float(data);
    val *= m_scale;
    return safetensors::float_to_bfloat16(val);
  } else {
    throw std::runtime_error("Unsupported LoRA dtype for scaling");
  }
}

void LoraContext::transpose_copy(uint16_t *dst, uint16_t *src, int rows,
                                 int cols, bool do_scale) {
  for (int r = 0; r < rows; r++) {
    uint16_t *src_row = src + r * cols;
    uint16_t *dst_col = dst + r;

    for (int c = 0; c < cols; c++) {
      if (!do_scale) {
        *dst_col = src_row[c];
      } else {
        *dst_col = a16_scale(src_row[c]);
      }
      dst_col += rows;
    }
  }
}

void LoraContext::transpose_stride_copy(uint16_t *dst, uint16_t *src, int dim0,
                                        int dim1, int max_dim1, bool do_scale) {
  for (int r = 0; r < dim0; r++) {
    uint16_t *src_row = src + r * dim1;

    for (int c = 0; c < dim1; c++) {
      uint16_t *dst_elem = dst + c * max_dim1 + r;
      if (!do_scale) {
        *dst_elem = src_row[c];
      } else {
        *dst_elem = a16_scale(src_row[c]);
      }
    }
  }
}

void LoraContext::stride_copy(uint16_t *dst, uint16_t *src, int dim0, int dim1,
                              int max_dim1, bool do_scale) {
  if (!do_scale) {
    for (int i = 0; i < dim0; i++) {
      uint16_t *src_row = src + i * dim1;
      uint16_t *dst_row = dst + i * max_dim1;
      std::memcpy(dst_row, src_row, dim1 * sizeof(uint16_t));
    }
  } else {
    for (int i = 0; i < dim0; i++) {
      uint16_t *src_row = src + i * dim1;
      uint16_t *dst_row = dst + i * max_dim1;
      for (int j = 0; j < dim1; j++) {
        dst_row[j] = a16_scale(src_row[j]);
      }
    }
  }
}

void LoraContext::data_copy(uint16_t *dst, uint16_t *src, size_t num_elems,
                            bool do_scale) {
  if (do_scale) {
    for (size_t i = 0; i < num_elems; i++) {
      dst[i] = a16_scale(src[i]);
    }
  } else {
    std::memcpy(dst, src, num_elems * sizeof(uint16_t));
  }
}