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
#ifndef UTILS_H_
#define UTILS_H_

#include "cnpy.h"
#include "tpuv7_modelrt.h"
#include <algorithm>
#include <climits>
#include <cmath>

static inline int getDtypeSize(const tpuRtDataType_t &dtype) {
  switch (dtype) {
  case TPU_FLOAT32:
  case TPU_INT32:
  case TPU_UINT32:
    return 4;
  case TPU_FLOAT16:
  case TPU_BFLOAT16:
  case TPU_INT16:
    return 2;
  case TPU_INT8:
  case TPU_UINT8:
    return 1;
  case TPU_INT4:
  case TPU_UINT4:
    return 1; // need modify ?  to do
  default:
    return 1;
  }
  return 0;
}

/* number of shape elements */
static inline uint64_t getShapeCount(const tpuRtShape_t &shape) {
  uint64_t count = 1;
  for (int i = 0; i < shape.num_dims; i++) {
    count *= shape.dims[i];
  }
  return count;
}

static void getInOutTensor(std::vector<tpuRtTensor_t> &in_tensors,
                           std::vector<tpuRtTensor_t> &out_tensors,
                           const tpuRtNetInfo_t &net, int stage_idx = 0) {
  auto mem = net.stages[stage_idx].input_mems;
  auto shape = net.stages[stage_idx].input_shapes;
  for (int i = 0; i < net.input.num; ++i) {
    in_tensors[i].dtype = net.input.dtypes[i];
    in_tensors[i].shape.num_dims = shape[i].num_dims;
    memcpy(in_tensors[i].shape.dims, shape[i].dims,
           sizeof(int) * shape[i].num_dims);
    in_tensors[i].data = mem[i];
  }
  mem = net.stages[stage_idx].output_mems;
  shape = net.stages[stage_idx].output_shapes;
  for (int i = 0; i < net.output.num; ++i) {
    out_tensors[i].dtype = net.output.dtypes[i];
    out_tensors[i].shape.num_dims = shape[i].num_dims;
    memcpy(out_tensors[i].shape.dims, shape[i].dims,
           sizeof(int) * shape[i].num_dims);
    out_tensors[i].data = mem[i];
  }
}

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

//===------------------------------------------------------------===//
// Dump Func
//===------------------------------------------------------------===//

void dump_tensor_to_file(tpuRtStream_t stream, const tpuRtTensor_t &t,
                         cnpy::npz_t &npz_map, const std::string &tensor_name) {
  auto tensor_type = t.dtype;
  int64_t mem_size = getShapeCount(t.shape) * getDtypeSize(t.dtype);
  std::vector<size_t> shape(t.shape.dims, t.shape.dims + t.shape.num_dims);
  if (tensor_type == TPU_FLOAT16) {
    // F16
    int cnt = mem_size / sizeof(uint16_t);
    std::vector<float> data(cnt);
    std::vector<uint16_t> buffer(cnt);
    tpuRtMemcpyD2S(buffer.data(), t.data, mem_size);
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = fp16_ieee_to_fp32_value(buffer[i]);
    }
    cnpy::npz_add_array(npz_map, tensor_name, data.data(), shape);
  } else if (tensor_type == TPU_BFLOAT16) {
    // BF16
    int cnt = mem_size / sizeof(uint16_t);
    std::vector<float> data(cnt);
    std::vector<uint16_t> buffer(cnt);
    tpuRtMemcpyD2S(buffer.data(), t.data, mem_size);
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = bf16_to_fp32_value(buffer[i]);
    }
    cnpy::npz_add_array(npz_map, tensor_name, data.data(), shape);
  } else if (tensor_type == TPU_INT32) {
    // INT32
    int cnt = mem_size / sizeof(int32_t);
    std::vector<int> data(cnt);
    tpuRtMemcpyD2S(data.data(), t.data, mem_size);
    cnpy::npz_add_array(npz_map, tensor_name, data.data(), shape);
  } else if (tensor_type == TPU_FLOAT32) {
    // FLOAT32
    int cnt = mem_size / sizeof(float);
    std::vector<float> data(cnt);
    tpuRtMemcpyD2S(data.data(), t.data, mem_size);
    cnpy::npz_add_array(npz_map, tensor_name, data.data(), shape);
  } else {
    throw std::runtime_error("Not support dtype");
  }
}

void dump_net_to_file(tpuRtStream_t stream, const tpuRtNetInfo_t &net,
                      const std::string &filename) {
  std::vector<tpuRtTensor_t> in_tensors(net.input.num);
  std::vector<tpuRtTensor_t> out_tensors(net.output.num);
  getInOutTensor(in_tensors, out_tensors, net);
  cnpy::npz_t npz_map;
  tpuRtStreamSynchronize(stream);
  for (int i = 0; i < net.input.num; i++) {
    dump_tensor_to_file(stream, in_tensors[i], npz_map, net.input.names[i]);
  }
  for (int i = 0; i < net.output.num; i++) {
    dump_tensor_to_file(stream, out_tensors[i], npz_map, net.output.names[i]);
  }
  cnpy::npz_save_all(filename, npz_map);
}

#endif // UTILS_H_