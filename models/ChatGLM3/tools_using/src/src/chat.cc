//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <bmruntime_interface.h>
#include <chat.h>
#include <sentencepiece_processor.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>
#include "memory.h"

static std::string TOKENIZER_MODEL;
static std::string CHATGLM_MODEL;
// #define EXPORT_RESULTS
#ifdef EXPORT_RESULTS
#include "cnpy.h"
static cnpy::npz_t map;

template <typename T>
static void add_array(
        std::string name,
        bm_handle_t bm_handle,
        const bm_device_mem_t& dst) {
    std::vector<T> data(dst.size / sizeof(T));
    bm_memcpy_d2s(bm_handle, data.data(), dst);
    cnpy::npz_add_array(map, name, data);
}

static void save_array(std::string filename) {
    cnpy::npz_save_all(filename, map);
}
#endif

void ChatGLM2::load_sentencepiece() {
    auto status = sentencepiece.Load(TOKENIZER_MODEL);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        exit(-1);
    }
    EOS = sentencepiece.eos_id();
}

void ChatGLM2::init(int devid, const char* model_dir) {
    printf("Initiating\n");
    TOKENIZER_MODEL = std::string(model_dir) + "/tokenizer.model";
    CHATGLM_MODEL = std::string(model_dir) + "/chatglm3-6b_int8.bmodel";

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
    ret = bmrt_tensor(
            &inputs_embed_512,
            p_bmrt,
            net_embed->input_dtypes[0],
            net_embed->stages[1].input_shapes[0]);
    assert(true == ret);

    ret = bmrt_tensor(
            &outputs_embed_512,
            p_bmrt,
            net_embed->output_dtypes[0],
            net_embed->stages[1].output_shapes[0]);
    assert(true == ret);

    ret = bmrt_tensor(
            &inputs_pid,
            p_bmrt,
            net_blocks[0]->input_dtypes[1],
            net_blocks[0]->stages[0].input_shapes[1]);
    assert(true == ret);

    ret = bmrt_tensor(
            &inputs_attention,
            p_bmrt,
            net_blocks[0]->input_dtypes[2],
            net_blocks[0]->stages[0].input_shapes[2]);
    assert(true == ret);

    ret = bmrt_tensor(
            &next_pid,
            p_bmrt,
            net_blocks_cache[0]->input_dtypes[1],
            net_blocks_cache[0]->stages[0].input_shapes[1]);
    assert(true == ret);

    ret = bmrt_tensor(
            &next_attention,
            p_bmrt,
            net_blocks_cache[0]->input_dtypes[2],
            net_blocks_cache[0]->stages[0].input_shapes[2]);
    assert(true == ret);

    for (int i = 0; i < NUM_LAYERS; i++) {
        ret = bmrt_tensor(
                &past_key[i],
                p_bmrt,
                net_blocks[0]->output_dtypes[1],
                net_blocks[0]->stages[0].output_shapes[1]);
        assert(true == ret);
        ret = bmrt_tensor(
                &past_value[i],
                p_bmrt,
                net_blocks[0]->output_dtypes[2],
                net_blocks[0]->stages[0].output_shapes[2]);
        assert(true == ret);
    }
    ret = bmrt_tensor(
            &inputs_lm,
            p_bmrt,
            net_lm->input_dtypes[0],
            net_lm->stages[0].input_shapes[0]);
    assert(true == ret);
    ret = bmrt_tensor(
            &outputs_lm,
            p_bmrt,
            net_lm->output_dtypes[0],
            net_lm->stages[0].output_shapes[0]);
    assert(true == ret);

    // std::cout << "Ready\n" << std::flush;
}

void ChatGLM2::deinit() {
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

void ChatGLM2::move2end(const bm_tensor_t& kv) {
    if (token_length >= MAX_LEN) {
        return;
    }
    auto total_size = bm_mem_get_device_size(kv.device_mem);
    auto bytes = total_size / MAX_LEN;
    auto real_size = token_length * bytes;
    auto mem = bm_mem_from_device(
            bm_mem_get_device_addr(kv.device_mem), real_size);
    auto buffer = new uint8_t[real_size];
    auto dst = new uint8_t[total_size];
    bm_memcpy_d2s(bm_handle, (void*)buffer, mem);
    memset(dst, 0, total_size - real_size);
    memcpy(dst + total_size - real_size, buffer, real_size);
    bm_memcpy_s2d(bm_handle, kv.device_mem, (void*)dst);
    delete[] buffer;
    delete[] dst;
}

int ChatGLM2::forward_first(std::vector<int>& tokens) {
    int input_ids[MAX_LEN] = {64790, 64792}; // start token
    int position_id[MAX_LEN] = {0};
    float attention_mask[MAX_LEN * MAX_LEN] = {0};

    // auto attention_mask = std::make_unique<float[]>(MAX_LEN * MAX_LEN);
    std::copy(tokens.begin(), tokens.end(), input_ids + 2);
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
    bm_memcpy_s2d(bm_handle, inputs_embed_512.device_mem, (void*)input_ids);
    // /std::cout << inputs_embed_512.device_mem.size << '\n';
    auto ret = bmrt_launch_tensor_ex(
            p_bmrt,
            name_embed.c_str(),
            &inputs_embed_512,
            1,
            &outputs_embed_512,
            1,
            true,
            false);
    assert(ret);
    bm_thread_sync(bm_handle);

    bm_memcpy_s2d(bm_handle, inputs_pid.device_mem, (void*)position_id);
    bm_memcpy_s2d(
            bm_handle, inputs_attention.device_mem, (void*)attention_mask);
    auto inputs_embed = outputs_embed_512;
    inputs_embed.shape = net_blocks[0]->stages[0].input_shapes[0];
    bm_tensor_t inputs_block[3] = {inputs_embed, inputs_pid, inputs_attention};
    for (int i = 0; i < NUM_LAYERS; i++) {
        bm_tensor_t outputs_block[3] = {
                inputs_embed, past_key[i], past_value[i]};
        ret = bmrt_launch_tensor_ex(
                p_bmrt,
                name_blocks[i].c_str(),
                inputs_block,
                3,
                outputs_block,
                3,
                true,
                false);
        assert(ret);
        bm_thread_sync(bm_handle);
        move2end(past_key[i]);
        move2end(past_value[i]);
    }
    int bytes = inputs_embed.device_mem.size / MAX_LEN;
    bm_memcpy_d2d_byte(
            bm_handle,
            inputs_lm.device_mem,
            0,
            inputs_embed.device_mem,
            (token_length - 1) * bytes,
            bytes);
    ret = bmrt_launch_tensor_ex(
            p_bmrt,
            name_lm.c_str(),
            &inputs_lm,
            1,
            &outputs_lm,
            1,
            true,
            false);
    bm_thread_sync(bm_handle);
    int token = 0;
    bm_memcpy_d2s(bm_handle, (void*)&token, outputs_lm.device_mem);

    return token;
}

int ChatGLM2::forward_next() {
    float attention_mask[MAX_LEN + 1] = {0};
    for (int i = 0; i <= MAX_LEN - token_length; i++) {
        attention_mask[i] = 1.0;
    }
    int32_t position_id = token_length - 1;
    // embedding
    outputs_lm.shape = net_embed->stages[0].input_shapes[0];
    auto ret = bmrt_launch_tensor_ex(
            p_bmrt,
            name_embed.c_str(),
            &outputs_lm,
            1,
            &inputs_lm,
            1,
            true,
            false);
    assert(ret);
    bm_thread_sync(bm_handle);

    bm_memcpy_s2d(bm_handle, next_attention.device_mem, (void*)attention_mask);
    bm_memcpy_s2d(bm_handle, next_pid.device_mem, (void*)&position_id);
    auto inputs_embed = inputs_lm;
    inputs_embed.shape = net_blocks_cache[0]->stages[0].input_shapes[0];
    for (int i = 0; i < NUM_LAYERS; i++) {
        bm_tensor_t inputs_block[5] = {
                inputs_embed,
                next_pid,
                next_attention,
                past_key[i],
                past_value[i]};
        bm_tensor_t outputs_block[3] = {
                inputs_embed, past_key[i], past_value[i]};
        ret = bmrt_launch_tensor_ex(
                p_bmrt,
                name_blocks_cache[i].c_str(),
                inputs_block,
                5,
                outputs_block,
                3,
                true,
                false);
        assert(ret);
        bm_thread_sync(bm_handle);
    }
    outputs_lm.shape = net_lm->stages[0].output_shapes[0];
    ret = bmrt_launch_tensor_ex(
            p_bmrt,
            name_lm.c_str(),
            &inputs_lm,
            1,
            &outputs_lm,
            1,
            true,
            false);
    bm_thread_sync(bm_handle);
    int token = 0;
    bm_memcpy_d2s(bm_handle, (void*)&token, outputs_lm.device_mem);

    return token;
}