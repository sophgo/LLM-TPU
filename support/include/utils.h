/*****************************************************************************
 *
 *    Copyright (c) 2016-2026 by Sophgo Technologies Inc. All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Sophgo Technologies Inc. This is proprietary information owned by
 *    Sophgo Technologies Inc. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Sophgo Technologies Inc.
 *
 *****************************************************************************/
#pragma once
#include "cnpy.h"
#include <algorithm>
#include <climits>
#include <cmath>

//===------------------------------------------------------------===//
// Union & Struct
//===------------------------------------------------------------===//
typedef union {
  uint16_t bits;
  struct {
    uint16_t frac : 10; // mantissa
    uint16_t exp : 5;   // exponent
    uint16_t sign : 1;  // sign
  } format;
} fp16;

typedef union {
  uint16_t bits;
  struct {
    uint16_t frac : 7; // mantissa
    uint16_t exp : 8;  // exponent
    uint16_t sign : 1; // sign
  } format;
} bf16;

typedef union {
  float fval;
  uint32_t bits;
  struct {
    uint32_t frac : 23; // mantissa
    uint32_t exp : 8;   // exponent
    uint32_t sign : 1;  // sign
  } format;
} fp32;

typedef struct {
  uint32_t magic;
  uint32_t header_size;
  uint32_t flatbuffers_size;
  uint32_t binary_size;
  uint32_t reserved[12];
} __attribute__((packed)) MODEL_HEADER_T;

//===------------------------------------------------------------===//
// Type Convert Func
//===------------------------------------------------------------===//
inline uint16_t fp32_to_fp16_bits(float f) {
  uint32_t x = *((uint32_t *)&f);
  uint16_t h = ((x >> 16) & 0x8000) |
               ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
               ((x >> 13) & 0x03ff);

  return h;
}

inline uint16_t fp32_to_bf16_bits(float f) {
  uint32_t x = *((uint32_t *)&f);
  uint16_t h = (x >> 16);

  return h;
}

static inline uint32_t bf16_to_fp32_bits(uint16_t h) {
  // BF16 的位模式是：1 位符号，8 位指数，7 位尾数
  // 我们需要将其转换为 float 的位模式：1 位符号，8 位指数，23 位尾数
  // 扩展 BF16 到 32 位，尾数部分需要填充 16 位的 0
  uint32_t sign = (uint32_t)(h & 0x8000) << 16; // 符号位
  uint32_t exp = (uint32_t)(h & 0x7F80) << 16;  // 指数位
  uint32_t frac = (uint32_t)(h & 0x007F) << 16; // 尾数位

  // 将尾数的 7 位左移，以对齐到 23 位尾数的位置
  // frac <<= (23 - 7);

  // 组合成 float 的位模式
  return sign | exp | frac;
}

static inline uint32_t fp16_ieee_to_fp32_bits(uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the bits 0-30
   * of the 32-bit word:
   *
   *      +---+-----+------------+-------------------+
   *      | 0 |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  30  27-31     17-26            0-16
   */
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  /*
   * Renorm shift is the number of bits to shift mantissa left to make the
   * half-precision number normalized. If the initial number is normalized, some
   * of its high 6 bits (sign == 0 and 5-bit exponent) equals one. In this case
   * renorm_shift == 0. If the number is denormalize, renorm_shift > 0. Note
   * that if we shift denormalized nonsign by renorm_shift, the unit bit of
   * mantissa will shift into exponent, turning the biased exponent into 1, and
   * making mantissa normalized (i.e. without leading 1).
   */
#ifdef _MSC_VER
  unsigned long nonsign_bsr;
  _BitScanReverse(&nonsign_bsr, (unsigned long)nonsign);
  uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
  uint32_t renorm_shift = __builtin_clz(nonsign);
#endif
  renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
  /*
   * Iff half-precision number has exponent of 15, the addition overflows it
   * into bit 31, and the subsequent shift turns the high 9 bits into 1. Thus
   *   inf_nan_mask ==
   *                   0x7F800000 if the half-precision number had exponent of
   * 15 (i.e. was NaN or infinity) 0x00000000 otherwise
   */
  const int32_t inf_nan_mask =
      ((int32_t)(nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
  /*
   * Iff nonsign is 0, it overflows into 0xFFFFFFFF, turning bit 31 into 1.
   * Otherwise, bit 31 remains 0. The signed shift right by 31 broadcasts bit 31
   * into all bits of the zero_mask. Thus zero_mask == 0xFFFFFFFF if the
   * half-precision number was zero (+0.0h or -0.0h) 0x00000000 otherwise
   */
  const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
  /*
   * 1. Shift nonsign left by renorm_shift to normalize it (if the input was
   * denormal)
   * 2. Shift nonsign right by 3 so the exponent (5 bits originally) becomes an
   * 8-bit field and 10-bit mantissa shifts into the 10 high bits of the 23-bit
   * mantissa of IEEE single-precision number.
   * 3. Add 0x70 to the exponent (starting at bit 23) to compensate the
   * different in exponent bias (0x7F for single-precision number less 0xF for
   * half-precision number).
   * 4. Subtract renorm_shift from the exponent (starting at bit 23) to account
   * for renormalization. As renorm_shift is less than 0x70, this can be
   * combined with step 3.
   * 5. Binary OR with inf_nan_mask to turn the exponent into 0xFF if the input
   * was NaN or infinity.
   * 6. Binary ANDNOT with zero_mask to turn the mantissa and exponent into zero
   * if the input was zero.
   * 7. Combine with the sign of the input number.
   */
  return sign |
         ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) |
           inf_nan_mask) &
          ~zero_mask);
}

float bf16_to_fp32_value(uint16_t d) {
  fp32 t;
  t.bits = bf16_to_fp32_bits(d);
  return t.fval;
}

float fp16_ieee_to_fp32_value(uint16_t d) {
  fp32 t;
  t.bits = fp16_ieee_to_fp32_bits(d);
  return t.fval;
}

uint16_t fp32_to_uint16(float value, bm_data_type_t tensor_type) {
  uint16_t uint16_value = 0;
  if (tensor_type == BM_FLOAT16) {
    uint16_value = fp32_to_fp16_bits(value);
  } else if (tensor_type == BM_BFLOAT16) {
    uint16_value = fp32_to_bf16_bits(value);
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }
  return uint16_value;
}

//===------------------------------------------------------------===//
// Dump Func
//===------------------------------------------------------------===//
float int_to_fp32(int int_val) { return static_cast<float>(int_val); }

float float_to_fp32(float float_val) { return float_val; }

void calculate_min_and_max(const std::vector<float> &data) {
  if (data.empty()) {
    std::cout << "No data to process." << std::endl;
    return;
  }
  auto min_it = std::min_element(data.begin(), data.end());
  auto max_it = std::max_element(data.begin(), data.end());
  std::cout << "min_value: " << *min_it << std::endl;
  std::cout << "max_value: " << *max_it << std::endl;
}

void dump_min_and_max_bf16(bm_handle_t bm_handle, bm_device_mem_t mem,
                           uint32_t (*converter)(uint16_t)) {
  size_t mem_size = bm_mem_get_device_size(mem);
  int ele_count = mem_size / sizeof(uint16_t);
  std::vector<uint16_t> data(ele_count);
  std::vector<float> fp32_data(ele_count);

  bm_memcpy_d2s_partial_offset(bm_handle, data.data(), mem, mem_size, 0);

  for (int i = 0; i < ele_count; i++) {
    fp32 t;
    t.bits = converter(data[i]);
    fp32_data[i] = t.fval;
  }

  calculate_min_and_max(fp32_data);
}

void dump_min_and_max_fp16(bm_handle_t bm_handle, bm_device_mem_t mem,
                           uint32_t (*converter)(uint16_t)) {
  size_t mem_size = bm_mem_get_device_size(mem);
  int ele_count = mem_size / sizeof(uint16_t);
  std::vector<uint16_t> data(ele_count);
  std::vector<float> fp32_data(ele_count);

  bm_memcpy_d2s_partial_offset(bm_handle, data.data(), mem, mem_size, 0);

  for (int i = 0; i < ele_count; i++) {
    fp32 t;
    t.bits = converter(data[i]);
    fp32_data[i] = t.fval;
  }

  calculate_min_and_max(fp32_data);
}

void dump_min_and_max_int(bm_handle_t bm_handle, bm_device_mem_t mem,
                          float (*converter)(int)) {
  size_t mem_size = bm_mem_get_device_size(mem);
  int ele_count = mem_size / sizeof(int);
  std::vector<int> data(ele_count);
  std::vector<float> fp32_data(ele_count);

  bm_memcpy_d2s_partial_offset(bm_handle, data.data(), mem, mem_size, 0);

  for (int i = 0; i < ele_count; i++) {
    fp32_data[i] = converter(data[i]);
  }

  calculate_min_and_max(fp32_data);
}

void dump_min_and_max_fp32(bm_handle_t bm_handle, bm_device_mem_t mem,
                           float (*converter)(float)) {
  size_t mem_size = bm_mem_get_device_size(mem);
  int ele_count = mem_size / sizeof(float);
  std::vector<float> data(ele_count);
  std::vector<float> fp32_data(ele_count);

  bm_memcpy_d2s_partial_offset(bm_handle, data.data(), mem, mem_size, 0);

  for (int i = 0; i < ele_count; i++) {
    fp32_data[i] = converter(data[i]);
  }

  calculate_min_and_max(fp32_data);
}

void dump_bf16_tensor(bm_handle_t bm_handle, bm_device_mem_t mem, int offset,
                      int size) {
  auto mem_size = bm_mem_get_device_size(mem);
  size = std::min(size, static_cast<int>(mem_size));
  int ele_count = size / sizeof(uint16_t);
  assert(mem_size < INT_MAX);

  std::vector<uint16_t> data(ele_count);
  bm_memcpy_d2s_partial_offset(bm_handle, data.data(), mem, size, offset);
  std::cout << "-------------------------------------" << std::endl;
  fp32 t;
  for (int i = 0; i < ele_count; i++) {
    t.bits = bf16_to_fp32_bits(data[i]);
    std::cout << t.fval << std::endl;
  }
  std::cout << "-------------------------------------" << std::endl;

  dump_min_and_max_bf16(bm_handle, mem, bf16_to_fp32_bits);
}

void dump_fp16_tensor(bm_handle_t bm_handle, bm_device_mem_t mem, int offset,
                      int size) {
  auto mem_size = bm_mem_get_device_size(mem);
  size = std::min(size, static_cast<int>(mem_size));
  int ele_count = size / sizeof(uint16_t);
  assert(mem_size < INT_MAX);

  std::vector<uint16_t> data(ele_count);
  bm_memcpy_d2s_partial_offset(bm_handle, data.data(), mem, size, offset);
  std::cout << "-------------------------------------" << std::endl;
  fp32 t;
  for (int i = 0; i < ele_count; i++) {
    t.bits = fp16_ieee_to_fp32_bits(data[i]);
    std::cout << t.fval << std::endl;
  }
  std::cout << "-------------------------------------" << std::endl;

  dump_min_and_max_fp16(bm_handle, mem, fp16_ieee_to_fp32_bits);
}

void dump_fp32_tensor(bm_handle_t bm_handle, bm_device_mem_t mem, int offset,
                      int size) {
  auto mem_size = bm_mem_get_device_size(mem);
  size = std::min(size, static_cast<int>(mem_size));
  int ele_count = size / sizeof(float);
  assert(mem_size < INT_MAX);

  std::vector<float> data(ele_count);
  bm_memcpy_d2s_partial_offset(bm_handle, data.data(), mem, size, offset);
  std::cout << "-------------------------------------" << std::endl;
  for (int i = 0; i < ele_count; i++) {
    std::cout << data[i] << std::endl;
  }
  std::cout << "-------------------------------------" << std::endl;

  dump_min_and_max_fp32(bm_handle, mem, float_to_fp32);
}

void dump_int_tensor(bm_handle_t bm_handle, bm_device_mem_t mem, int offset,
                     int size) {
  auto mem_size = bm_mem_get_device_size(mem);
  size = std::min(size, static_cast<int>(mem_size));
  int ele_count = size / sizeof(int);
  assert(mem_size < INT_MAX);

  std::vector<int> data(ele_count);
  bm_memcpy_d2s_partial_offset(bm_handle, data.data(), mem, size, offset);
  std::cout << "-------------------------------------" << std::endl;
  for (int i = 0; i < ele_count; i++) {
    std::cout << data[i] << std::endl;
  }
  std::cout << "-------------------------------------" << std::endl;

  dump_min_and_max_int(bm_handle, mem, int_to_fp32);
}

//===------------------------------------------------------------===//
// Calculate the similarity
//===------------------------------------------------------------===//

std::vector<float> vec_bf16_to_fp32(const std::vector<uint16_t> &tar) {
  std::vector<float> data(tar.size(), 0);
  for (size_t i = 0; i < tar.size(); ++i)
    data[i] = bf16_to_fp32_value(tar[i]);
  return data;
}

std::vector<float> vec_fp16_to_fp32(const std::vector<uint16_t> &tar) {
  std::vector<float> data(tar.size(), 0);
  for (size_t i = 0; i < tar.size(); ++i)
    data[i] = fp16_ieee_to_fp32_value(tar[i]);
  return data;
}

std::vector<float> vec_int_to_fp32(const std::vector<int> &tar) {
  std::vector<float> data(tar.size(), 0);
  for (size_t i = 0; i < tar.size(); ++i)
    data[i] = static_cast<float>(tar[i]);
  return data;
}

void cal_similarity(const std::vector<float> &data,
                    std::vector<float> &ref_data) {
  std::cout << "-------------------------------------" << std::endl;
  if (data.size() != ref_data.size()) {
    throw std::invalid_argument("The sizes of data and ref_data do not match.");
  }
  std::vector<float> noise(data.size(), 0);

  float distance = 0, root = 0, L1_distance = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    noise[i] = ref_data[i] - data[i];
    distance += pow(data[i] - ref_data[i], 2);
    root += pow((data[i] + ref_data[i]) / 2, 2);
    L1_distance += abs(data[i] - ref_data[i]);
  }
  distance = sqrt(distance);
  root = sqrt(root);
  std::cout << "    manhattan_distance   = " << (float)(L1_distance)
            << std::endl;

  std::cout << "    euclidean_similarity   = " << (float)(1 - distance / root)
            << std::endl;

  float average = 0, ss_tar = 0, ss_ref = 0, avg_ref = 0, avg_noise = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    average += data[i] * ref_data[i];
    ss_tar += pow(data[i], 2);
    ss_ref += pow(ref_data[i], 2);
    avg_ref += ref_data[i];
    avg_noise += noise[i];
  }
  average /= data.size();
  ss_tar /= data.size();
  ss_ref /= data.size();
  avg_ref /= data.size();
  avg_noise /= data.size();
  std::cout << "    cosine_similarity      = "
            << (average / sqrt(ss_tar * ss_ref)) << std::endl;

  float var_ref_zero_mean = 0, var_noise_zero_mean = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    var_ref_zero_mean += pow(ref_data[i] - avg_ref, 2);
    var_noise_zero_mean += pow(noise[i] - avg_noise, 2);
  }
  float sqnr = 0;
  if (var_ref_zero_mean == 0.0 || var_noise_zero_mean == 0.0) {
    sqnr = std::numeric_limits<float>::infinity();
  } else {
    sqnr = 10 * std::log10(var_ref_zero_mean / var_noise_zero_mean);
  }
  std::cout << "    sqnr_similarity        = " << sqnr << std::endl;
  std::cout << "-------------------------------------" << std::endl;
}

void compare_bf16_similarity(bm_handle_t &bm_handle, bm_device_mem_t &mem,
                             std::vector<float> &ref_data) {
  int mem_size = bm_mem_get_device_size(mem);
  int cnt = mem_size / sizeof(uint16_t);
  std::vector<uint16_t> buffer(cnt);
  bm_memcpy_d2s(bm_handle, buffer.data(), mem);
  std::vector<float> data = vec_bf16_to_fp32(buffer);
  cal_similarity(data, ref_data);
}

void compare_fp16_similarity(bm_handle_t &bm_handle, bm_device_mem_t &mem,
                             std::vector<float> &ref_data) {
  int mem_size = bm_mem_get_device_size(mem);
  int cnt = mem_size / sizeof(uint16_t);
  std::vector<uint16_t> buffer(cnt);
  bm_memcpy_d2s(bm_handle, buffer.data(), mem);
  std::vector<float> data = vec_fp16_to_fp32(buffer);
  cal_similarity(data, ref_data);
}

void compare_int_similarity(bm_handle_t &bm_handle, bm_device_mem_t &mem,
                            std::vector<float> &ref_data) {
  int mem_size = bm_mem_get_device_size(mem);
  int cnt = mem_size / sizeof(int32_t);
  std::vector<int> buffer(cnt);
  bm_memcpy_d2s(bm_handle, buffer.data(), mem);
  std::vector<float> data = vec_int_to_fp32(buffer);
  cal_similarity(data, ref_data);
}

void compare_fp32_similarity(bm_handle_t &bm_handle, bm_device_mem_t &mem,
                             std::vector<float> &ref_data) {
  int mem_size = bm_mem_get_device_size(mem);
  int cnt = mem_size / sizeof(float);
  std::vector<float> data(cnt);
  bm_memcpy_d2s(bm_handle, data.data(), mem);
  cal_similarity(data, ref_data);
}

void compare_similarity(bm_handle_t &bm_handle, bm_device_mem_t &mem,
                        bm_data_type_t &tensor_type, std::string v_file,
                        std::string v_name) {
  cnpy::NpyArray ref_file = cnpy::npz_load(v_file, v_name);
  std::vector<float> ref_data = ref_file.as_vec<float>();

  if (tensor_type == BM_FLOAT16) {
    compare_fp16_similarity(bm_handle, mem, ref_data);
  } else if (tensor_type == BM_BFLOAT16) {
    compare_bf16_similarity(bm_handle, mem, ref_data);
  } else if (tensor_type == BM_INT32) {
    compare_int_similarity(bm_handle, mem, ref_data);
  } else if (tensor_type == BM_FLOAT32) {
    compare_int_similarity(bm_handle, mem, ref_data);
  }
}

void compare_in_net(
    bm_handle_t &bm_handle, const bm_net_info_t *net, std::string filename,
    const std::vector<std::string> &names = std::vector<std::string>()) {
  std::vector<bm_tensor_t> tensors(net->output_num);

  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&tensors[i], net->stages[0].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[0].output_shapes[i]);
    if (names.size() == 0) {
      compare_similarity(bm_handle, tensors[i].device_mem, net->input_dtypes[i],
                         filename, "input_" + std::to_string(i));
    } else {
      compare_similarity(bm_handle, tensors[i].device_mem, net->input_dtypes[i],
                         filename, names[i]);
    }
  }
}

void compare_out_net(
    bm_handle_t &bm_handle, const bm_net_info_t *net, std::string filename,
    const std::vector<std::string> &names = std::vector<std::string>()) {
  std::vector<bm_tensor_t> tensors(net->output_num);

  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&tensors[i], net->stages[0].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[0].output_shapes[i]);
    if (names.size() == 0) {
      compare_similarity(bm_handle, tensors[i].device_mem, net->input_dtypes[i],
                         filename, "output_" + std::to_string(i));
    } else {
      compare_similarity(bm_handle, tensors[i].device_mem, net->input_dtypes[i],
                         filename, names[i]);
    }
  }
}

//===------------------------------------------------------------===//
// Dump to file
//===------------------------------------------------------------===//
void dump_tensor_to_file(bm_handle_t &bm_handle, bm_tensor_t &t,
                         bm_shape_t bm_shape, const std::string &filename,
                         bm_data_type_t tensor_type,
                         const std::string &tensor_name) {
  int mem_size = bm_mem_get_device_size(t.device_mem);
  std::vector<size_t> shape(bm_shape.dims, bm_shape.dims + bm_shape.num_dims);
  if (tensor_type == BM_FLOAT16) {
    // F16
    int cnt = mem_size / sizeof(uint16_t);
    std::vector<float> data(cnt);
    std::vector<uint16_t> buffer(cnt);
    bm_memcpy_d2s(bm_handle, buffer.data(), t.device_mem);
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = fp16_ieee_to_fp32_value(buffer[i]);
    }
    cnpy::npz_save(filename, tensor_name, data.data(), shape, "a");
  } else if (tensor_type == BM_BFLOAT16) {
    // BF16
    int cnt = mem_size / sizeof(uint16_t);
    std::vector<float> data(cnt);
    std::vector<uint16_t> buffer(cnt);
    bm_memcpy_d2s(bm_handle, buffer.data(), t.device_mem);
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = bf16_to_fp32_value(buffer[i]);
    }
    cnpy::npz_save(filename, tensor_name, data.data(), shape, "a");
  } else if (tensor_type == BM_INT32) {
    // INT32
    int cnt = mem_size / sizeof(int32_t);
    std::vector<int> data(cnt);
    bm_memcpy_d2s(bm_handle, data.data(), t.device_mem);
    cnpy::npz_save(filename, tensor_name, data.data(), shape, "a");
  } else if (tensor_type == BM_FLOAT32) {
    // FLOAT32
    int cnt = mem_size / sizeof(float);
    std::vector<float> data(cnt);
    bm_memcpy_d2s(bm_handle, data.data(), t.device_mem);
    cnpy::npz_save(filename, tensor_name, data.data(), shape, "a");
  } else {
    throw std::runtime_error("Not support dtype");
  }
}

void dump_net_input_to_file(bm_handle_t &bm_handle, const bm_net_info_t *net,
                            const std::string &filename) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);

  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[0].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[0].input_shapes[i]);

    dump_tensor_to_file(bm_handle, in_tensors[i],
                        net->stages[0].input_shapes[i], filename,
                        net->input_dtypes[i], "input_" + std::to_string(i));
  }
}

void dump_net_output_to_file(bm_handle_t &bm_handle, const bm_net_info_t *net,
                             const std::string &filename) {
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net->stages[0].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[0].output_shapes[i]);

    dump_tensor_to_file(bm_handle, out_tensors[i],
                        net->stages[0].output_shapes[i], filename,
                        net->output_dtypes[i], "output_" + std::to_string(i));
  }
}

void dump_net_to_file(bm_handle_t &bm_handle, const bm_net_info_t *net,
                      const std::string &filename) {
  dump_net_input_to_file(bm_handle, net, filename);
  dump_net_output_to_file(bm_handle, net, filename);
}

//===------------------------------------------------------------===//
// Empty Func
//===------------------------------------------------------------===//
void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
}

void empty_in_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
                  int stage_idx = 0) {
  for (int i = 0; i < net->input_num; i++) {
    empty(bm_handle, net->stages[stage_idx].input_mems[i]);
  }
}

void empty_out_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
                   int stage_idx = 0) {
  for (int i = 0; i < net->output_num; i++) {
    empty(bm_handle, net->stages[stage_idx].output_mems[i]);
  }
}

void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
               int stage_idx = 0) {
  empty_in_net(bm_handle, net, stage_idx);
  empty_out_net(bm_handle, net, stage_idx);
}

//===------------------------------------------------------------===//
// Config
//===------------------------------------------------------------===//
struct Config {
  std::string model_type;
  int SEQLEN;
  int MAX_PREFILL_LENGTH;
  int total_length;
  uint16_t mask_value;

  // vit config
  int max_pos;
  int MAX_PIXELS;
  std::vector<int> grid_thw;
  int vit_offset;
  int valid_vit_length;
  int spatial_merge_size;
};

//===------------------------------------------------------------===//
// Make ViT position_id & attention_mask
//===------------------------------------------------------------===//
std::vector<int> make_vit_position_id(const Config &config) {
  std::vector<int> pos_ids;
  if (config.model_type == "qwen2_vl") {
    int t = config.grid_thw[0];
    int h = config.grid_thw[1];
    int w = config.grid_thw[2];

    // generate hpos_ids
    std::vector<int> hpos_ids;
    for (int n = 0; n < h; n += config.spatial_merge_size) {
      for (int _ = 0; _ < w / config.spatial_merge_size; ++_) {
        hpos_ids.push_back(n);
        hpos_ids.push_back(n);
        hpos_ids.push_back(n + 1);
        hpos_ids.push_back(n + 1);
      }
    }

    // generate wpos_ids
    std::vector<int> wpos_ids;
    for (int _ = 0; _ < h / config.spatial_merge_size; ++_) {
      for (int e = 0; e < w; e += config.spatial_merge_size) {
        wpos_ids.push_back(e);
        wpos_ids.push_back(e + 1);
        wpos_ids.push_back(e);
        wpos_ids.push_back(e + 1);
      }
    }

    int valid_vit_pixels = h * w;
    pos_ids.resize(config.MAX_PIXELS * 2, 0);
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < valid_vit_pixels; ++j) {
        pos_ids[i * valid_vit_pixels + 2 * j] = hpos_ids[j];
        pos_ids[i * valid_vit_pixels + 2 * j + 1] = wpos_ids[j];
      }
    }
  } else {
    throw std::runtime_error("not support now");
  }

  return pos_ids;
}

std::vector<uint16_t> make_vit_attention_mask(const Config &config) {
  std::vector<uint16_t> attention_mask;
  if (config.model_type == "qwen2_vl") {
    // Extract t, h, w from config.grid_thw
    int t = config.grid_thw[0];
    int h = config.grid_thw[1];
    int w = config.grid_thw[2];

    // Compute cu_seqlens
    std::vector<int> cu_seqlens(t + 1, 0);
    for (int i = 0; i <= t; ++i) {
      cu_seqlens[i] = h * w * i;
    }

    // Initialize attention_mask with -10000
    attention_mask.resize(config.MAX_PIXELS * config.MAX_PIXELS,
                          config.mask_value);

    // Update attention_mask based on cu_seqlens
    for (size_t i = 1; i < cu_seqlens.size(); ++i) {
      int start = cu_seqlens[i - 1];
      int end = cu_seqlens[i];
      for (int row = start; row < end; ++row) {
        for (int col = start; col < end; ++col) {
          size_t index = row * config.MAX_PIXELS + col;
          if (index < attention_mask.size()) {
            attention_mask[index] = 0;
          }
        }
      }
    }
  } else {
    throw std::runtime_error("not support now");
  }

  return attention_mask;
}

//===------------------------------------------------------------===//
// Make LLM position_id & attention_mask (Prefill Phase)
//===------------------------------------------------------------===//
std::vector<int> make_position_id(Config &config) {
  std::vector<int> position_id;
  if (config.model_type == "qwen2_vl") {
    int text_len = config.vit_offset;

    // Assuming config.grid_thw has at least one element
    int llm_grid_t = config.grid_thw[0];
    int llm_grid_h = config.grid_thw[1] / config.spatial_merge_size;
    int llm_grid_w = config.grid_thw[2] / config.spatial_merge_size;

    std::vector<int> t_position_id;
    std::vector<int> h_position_id;
    std::vector<int> w_position_id;

    // Populate t_position_id
    for (int i = text_len; i < llm_grid_t + text_len; ++i) {
      for (int j = 0; j < llm_grid_h * llm_grid_w; ++j) {
        t_position_id.push_back(i);
      }
    }

    // Populate h_position_id
    for (int _ = 0; _ < llm_grid_t; ++_) {
      for (int i = 0; i < llm_grid_h; ++i) {
        for (int j = 0; j < llm_grid_w; ++j) {
          h_position_id.push_back(i + text_len);
        }
      }
    }

    // Populate w_position_id
    for (int _ = 0; _ < llm_grid_t; ++_) {
      for (int i = 0; i < llm_grid_h; ++i) {
        for (int j = text_len; j < llm_grid_w + text_len; ++j) {
          w_position_id.push_back(j);
        }
      }
    }

    // Calculate starting index for tail text length
    int st_idx = w_position_id.back() + 1;
    int tail_text_len =
        config.total_length - config.valid_vit_length - text_len;

    // Prepare final position ids
    position_id.reserve(config.SEQLEN * 3);

    // Prepare head position ids
    std::vector<int> head_position_id;
    for (int i = 0; i < text_len; ++i) {
      head_position_id.push_back(i);
    }

    // Prepare tail position ids
    std::vector<int> tail_position_id;
    for (int i = st_idx; i < st_idx + tail_text_len; ++i) {
      tail_position_id.push_back(i);
    }

    // Fill position_id for t
    position_id.insert(
        position_id.end(), head_position_id.begin(),
        head_position_id.end()); // Fill with 0 for range text_len
    position_id.insert(position_id.end(), t_position_id.begin(),
                       t_position_id.end());
    position_id.insert(position_id.end(), tail_position_id.begin(),
                       tail_position_id.end());
    position_id.insert(position_id.end(), config.SEQLEN - config.total_length,
                       1); // Fill with 1

    // Fill position_id for h
    position_id.insert(
        position_id.end(), head_position_id.begin(),
        head_position_id.end()); // Fill with 0 for range text_len
    position_id.insert(position_id.end(), h_position_id.begin(),
                       h_position_id.end());
    position_id.insert(position_id.end(), tail_position_id.begin(),
                       tail_position_id.end());
    position_id.insert(position_id.end(), config.SEQLEN - config.total_length,
                       1); // Fill with 1

    // Fill position_id for w
    position_id.insert(
        position_id.end(), head_position_id.begin(),
        head_position_id.end()); // Fill with 0 for range text_len
    position_id.insert(position_id.end(), w_position_id.begin(),
                       w_position_id.end());
    position_id.insert(position_id.end(), tail_position_id.begin(),
                       tail_position_id.end());
    position_id.insert(position_id.end(), config.SEQLEN - config.total_length,
                       1); // Fill with 1

    config.max_pos = st_idx + tail_text_len - 1;
  } else {
    position_id.resize(config.MAX_PREFILL_LENGTH, 0);
    for (int i = 0; i < config.total_length; i++) {
      position_id[i] = i;
    }
  }

  return position_id;
}

std::vector<uint16_t> make_attention_mask(const Config &config) {
  std::vector<uint16_t> attention_mask(
      config.MAX_PREFILL_LENGTH * config.MAX_PREFILL_LENGTH, config.mask_value);
  for (int i = 0; i < config.total_length; i++) {
    for (int j = 0; j < config.total_length; j++) {
      if (j <= i) {
        attention_mask[i * config.MAX_PREFILL_LENGTH + j] = 0;
      }
    }
  }

  return attention_mask;
}

//===------------------------------------------------------------===//
// Make LLM position_id & attention_mask (Decode Phase)
//===------------------------------------------------------------===//
std::vector<int> make_next_position_id(Config &config) {
  std::vector<int> position_id;
  if (config.model_type == "qwen2_vl") {
    config.max_pos += 1;
    position_id = {config.max_pos, config.max_pos, config.max_pos};
  } else {
    position_id = {config.total_length - 1};
  }

  return position_id;
}