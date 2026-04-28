//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

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
#include <random>
#include <stdio.h>
#include <vector>

typedef std::vector<int> ArrayInt;
typedef std::vector<float> ArrayFloat;
typedef std::vector<std::vector<int>> ArrayInt2D;
typedef std::vector<std::vector<float>> ArrayFloat2D;
typedef std::vector<uint16_t> ArrayUint16;
typedef std::vector<std::vector<uint16_t>> ArrayUint162D;

void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem);
void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net, int stage = 0);

void init_tensors(const bm_net_info_t *net,
                  std::vector<bm_tensor_t> &in_tensors,
                  std::vector<bm_tensor_t> &out_tensors, int stage = 0);

void net_launch(void *p_bmrt, const bm_net_info_t *net,
                const std::vector<bm_tensor_t> &in_tensors,
                std::vector<bm_tensor_t> &out_tensors);

void d2d(bm_handle_t &bm_handle, bm_device_mem_t &dst, bm_device_mem_t &src,
         int offset = 0, int size = 0);