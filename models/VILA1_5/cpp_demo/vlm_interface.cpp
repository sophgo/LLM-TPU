#include "vlm_interface.hpp"
#include "PillowResize.hpp"
#include "bmruntime_interface.h"
#include "sentencepiece/sentencepiece_processor.h"
#include <iostream>
#include <vector>

static const int IMG_TOKEN = -200;
static const int IMG_HEIGHT = 384;
static const int IMG_WIDTH = 384;
static const bool ONLY_FIRST_OUTPUT = true;

class Vila {
public:
  Vila() { is_init = false; }
  void init(const std::string &llm_model, const std::string &vit_model,
            const std::string &tokenizer_path, int devid = 0);
  void deinit();
  void tokenizer_image_token(std::vector<int> &input_ids,
                             const std::string &prompt,
                             int image_token_index = IMG_TOKEN,
                             bool lstrip = false);
  std::string tokenizer_decode(std::vector<int> &tokens);
  void vit_inference(const std::vector<float> &pixel_values,
                     std::vector<fp16> &out_feature);
  void forward_first(const std::vector<int> &tokens, const HalfMatrix &features,
                     int &token, float &score);
  void forward_next(int &token, float &score);

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);

public:
  int token_length;
  int SEQLEN; // read from bmodel
  int HIDDEN_SIZE;
  int NUM_LAYERS; // read from bmodel
  int VISION_TOKEN_LEN;
  int SOS;
  int EOS;
  uint64_t IMG_BYTES;
  bool is_init;
  bool support_score;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_vit;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
  sentencepiece::SentencePieceProcessor sentencepiece;
};

static const uint16_t ATTENTION_MASK = 0xF0E2;

void Vila::net_launch(const bm_net_info_t *net, int stage_idx) {
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
  assert(ret);
 // bm_thread_sync(bm_handle);
}

void Vila::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Vila::init(const std::string &llm_model, const std::string &vit_model,
                const std::string &tokenizer_path, int devid) {
  assert(is_init == false);
  // request bm_handle
  std::cout << "Device [ " << devid << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, devid);
  assert(BM_SUCCESS == status);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("LLM Model[%s] loading ....\n", llm_model.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, llm_model.c_str());
  assert(true == ret);
  printf("Done!\n");
  printf("VIT Model[%s] loading ....\n", vit_model.c_str());
  ret = bmrt_load_bmodel(p_bmrt, vit_model.c_str());
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_vit = bmrt_get_network_info(p_bmrt, "vision_embedding");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  support_score = net_lm->output_num == 2;
  SEQLEN = net_embed->stages[0].output_shapes[0].dims[1]; // real seqlen
  HIDDEN_SIZE = net_embed->stages[0].output_shapes[0].dims[2];
  VISION_TOKEN_LEN = net_vit->stages[0].output_shapes[0].dims[1];
  IMG_BYTES = bm_mem_get_device_size(net_vit->stages[0].input_mems[0]);
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 4) / 2;
  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
  }

  // load sentencepiece
  auto st_ret = sentencepiece.Load(tokenizer_path);
  if (!st_ret.ok()) {
    std::cout << st_ret.ToString() << std::endl;
    exit(-1);
  }
  SOS = sentencepiece.bos_id();
  EOS = sentencepiece.eos_id();
  is_init = true;
}

void Vila::deinit() {
  assert(is_init == true);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
  is_init = false;
}

std::string Vila::tokenizer_decode(std::vector<int> &tokens) {
  std::string output;
  sentencepiece.Decode(tokens, &output);
  return output;
}

static void string_split(std::vector<std::string> &res, const std::string &str,
                         const std::string &splits) {
  if (str == "")
    return;

  // add splits to end
  std::string strs = str + splits;
  size_t pos = strs.find(splits);
  int step = splits.size();

  while (pos != strs.npos) {
    std::string temp = strs.substr(0, pos);
    res.push_back(temp);
    // delete
    strs = strs.substr(pos + step, strs.size());
    pos = strs.find(splits);
  }
}

void Vila::tokenizer_image_token(std::vector<int> &input_ids,
                                 const std::string &prompt,
                                 int image_token_index, bool lstrip) {
  // token2id
  std::vector<std::string> split_strings;
  std::vector<std::vector<int>> prompt_chunks;
  string_split(split_strings, prompt, "<image>");
  for (auto &split_string : split_strings) {
    split_string = " " + split_string;
    std::vector<int> chunk;
    sentencepiece.Encode(split_string, &chunk);
    chunk.insert(chunk.begin(), SOS);
    prompt_chunks.push_back(chunk);
  }

  int offset = 0;
  if (lstrip) {
    offset = 1;
  } else {
    if (prompt_chunks.size() > 0 && prompt_chunks[0].size() > 0 &&
        prompt_chunks[0][0] == SOS) {
      offset = 1;
      input_ids.push_back(prompt_chunks[0][0]);
    }
  }

  std::vector<int> sep(offset + 1, image_token_index);
  std::vector<std::vector<int>> insert_separator;
  for (uint32_t chunk_id = 0; chunk_id < prompt_chunks.size(); chunk_id++) {
    insert_separator.push_back(prompt_chunks[chunk_id]);
    if (chunk_id != prompt_chunks.size() - 1) {
      insert_separator.push_back(sep);
    }
  }
  for (uint32_t chunk_id = 0; chunk_id < insert_separator.size(); chunk_id++) {
    if (chunk_id == 0 && lstrip) {
      for (uint32_t j = 0; j < insert_separator[chunk_id].size(); j++) {
        input_ids.push_back(insert_separator[chunk_id][j]);
      }
    } else {
      for (uint32_t j = offset; j < insert_separator[chunk_id].size(); j++) {
        input_ids.push_back(insert_separator[chunk_id][j]);
      }
    }
  }
}

void Vila::vit_inference(const std::vector<float> &pixel_values,
                         std::vector<fp16> &out_feature) {
  if (pixel_values.size() * sizeof(float) != IMG_BYTES) {
    std::cout << "Input pixel size is not correct!" << std::endl;
    exit(-1);
  }
  auto &vit_in_mem = net_vit->stages[0].input_mems[0];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, vit_in_mem, (void *)pixel_values.data());
  net_launch(net_vit);
  out_feature.resize(VISION_TOKEN_LEN * HIDDEN_SIZE);
  bm_memcpy_d2s(bm_handle, (void *)out_feature.data(), vit_out_mem);
}

void Vila::forward_first(const std::vector<int> &tokens,
                         const HalfMatrix &features, int &token, float &score) {
  std::vector<int> input_ids(SEQLEN, 0);
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  uint32_t num_frames = features.size();
  token_length = tokens.size() + num_frames * (VISION_TOKEN_LEN - 1);
  std::vector<int> image_offset_v;
  int offset = 0;
  int last = 0;
  for (uint32_t i = 0; i < tokens.size(); i++) {
    if (tokens[i] == IMG_TOKEN) {
      // std::cout << "index:" << i << std::endl;
      std::copy(tokens.begin() + last, tokens.begin() + i,
                input_ids.begin() + offset);
      int image_offset = offset + i - last;
      image_offset_v.push_back(image_offset);
      offset = image_offset + VISION_TOKEN_LEN;
      last = i + 1;
    } else if (i + 1 == tokens.size()) {
      // std::cout << "last index:" << i << std::endl;
      std::copy(tokens.begin() + last, tokens.end(),
                input_ids.begin() + offset);
      int len = i + 1 - last + offset;
      assert(len == token_length);
    }
  }
  assert(num_frames == image_offset_v.size());

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < token_length; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  auto in_mem = net_embed->stages[0].input_mems[0];
  auto out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)input_ids.data());
  net_launch(net_embed); // prefil embedding

  auto &in0_mem = net_blocks[0]->stages[0].input_mems[0];
  d2d(in0_mem, out_mem);
  if (num_frames > 0) {
    uint64_t size = VISION_TOKEN_LEN * HIDDEN_SIZE * sizeof(fp16);
    uint64_t addr = bm_mem_get_device_addr(in0_mem);
    for (uint32_t i = 0; i < num_frames; i++) {
      bm_device_mem_t dst_mem;
      uint64_t device_addr =
          addr + image_offset_v[i] * HIDDEN_SIZE * sizeof(fp16);
      bm_set_device_mem(&dst_mem, size, device_addr);
      bm_memcpy_s2d(bm_handle, dst_mem, (void *)features[i].data());
    }
  }

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    if (idx == 0) {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    } else {
      d2d(in0_mem, out_mem);
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
  }

  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * bytes, bytes);
  net_launch(net_lm);
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  if (support_score) {
    auto &lm_score = net_lm->stages[0].output_mems[1];
    bm_memcpy_d2s(bm_handle, (void *)&score, lm_score);
  } else {
    score = 0.0f;
  }
  token_length++;
}

void Vila::forward_next(int &token, float &score) {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;

  // embedding
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  auto in_mem = net_embed_cache->stages[0].input_mems[0];
  auto out_mem = net_embed_cache->stages[0].output_mems[0];
  d2d(in_mem, lm_out_mem);
  net_launch(net_embed_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (token_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    } else {
      d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
      d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
    }

    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       bytes);
  }

  // forward lmhead
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm);
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  if (support_score) {
    auto &lm_score = net_lm->stages[0].output_mems[1];
    bm_memcpy_d2s(bm_handle, (void *)&score, lm_score);
  }
  token_length++;
}

// interface implement

static Vila g_vila;

void initVLMModel(const std::string &llm_bmodel, const std::string &vit_bmodel,
                  const std::string &tokenizer_path) {
  g_vila.init(llm_bmodel, vit_bmodel, tokenizer_path);
}

void destroyVLMModel() { g_vila.deinit(); }

// void extractVLMVisualFeature(const std::vector<cv::Mat> &images,
//                              HalfMatrix &output_features) {
//   for (auto image : images) {
//     // preprocess
//     bm_image imagein, imageout;
//     bmcv_convert_to_attr convertto_cfg;
//     cv::bmcv::toBMI(image, &imagein);
//     bm_handle_t handle = bm_image_get_handle(&imagein);
//     bm_status_t ret = BM_SUCCESS;
//     bm_device_mem_t dmem[3];
//     unsigned long long vir_addr[3];
//     bm_image_create(handle, IMG_HEIGHT, IMG_WIDTH, FORMAT_BGRP_SEPARATE,
//                     DATA_TYPE_EXT_FLOAT32, &imageout);
//     bm_image_alloc_dev_mem(imageout, BMCV_HEAP1_ID);

//     convertto_cfg.alpha_0 = 0.00392156862745098 * 1.0 / 0.5;
//     convertto_cfg.alpha_1 = 0.00392156862745098 * 1.0 / 0.5;
//     convertto_cfg.alpha_2 = 0.00392156862745098 * 1.0 / 0.5;
//     convertto_cfg.beta_0 = (0.0 - 0.5) / 0.5;
//     convertto_cfg.beta_1 = (0.0 - 0.5) / 0.5;
//     convertto_cfg.beta_2 = (0.0 - 0.5) / 0.5;
//     ret = bmcv_image_csc_convert_to(handle, 1, &imagein, &imageout, NULL,
//     NULL,
//                                     NULL, BMCV_INTER_LINEAR, CSC_MAX_ENUM,
//                                     NULL, &convertto_cfg);
//     if (ret != BM_SUCCESS) {
//       printf("bmcv fail, ret(%d)\n", ret);
//       exit(-1);
//     }

//     bm_image_get_device_mem(imageout, dmem);

//     bm_mem_mmap_device_mem(handle, &dmem[0], &vir_addr[0]);
//     bm_mem_mmap_device_mem(handle, &dmem[1], &vir_addr[1]);
//     bm_mem_mmap_device_mem(handle, &dmem[2], &vir_addr[2]);

//     // convert to array
//     const int image_area = IMG_HEIGHT * IMG_WIDTH;
//     size_t single_chn_size = image_area * sizeof(float);
//     std::vector<float> process_image;
//     process_image.resize(image_area * 3);
//     memcpy((void *)process_image.data(), (float *)vir_addr[0],
//     single_chn_size); memcpy((void *)(process_image.data() + image_area),
//     (float *)vir_addr[1],
//            single_chn_size);
//     memcpy((void *)(process_image.data() + image_area * 2),
//            (float *)vir_addr[2], single_chn_size);

//     bm_mem_unmap_device_mem(handle, (void *)vir_addr[0], single_chn_size);
//     bm_mem_unmap_device_mem(handle, (void *)vir_addr[1], single_chn_size);
//     bm_mem_unmap_device_mem(handle, (void *)vir_addr[2], single_chn_size);
//     bm_image_destroy(imageout);

//     output_features.push_back(process_image);
//   }
// }

void extractVLMVisualFeature(const std::vector<cv::Mat> &images,
                             HalfMatrix &output_features) {
  assert(g_vila.is_init == true);
  const int image_area = IMG_HEIGHT * IMG_WIDTH;
  size_t single_chn_size = image_area * sizeof(float);
  std::vector<float> process_image(image_area * 3);
  for (auto &image : images) {
    // preprocess
    auto resized_img =
        PillowResize::resize(image, cv::Size(IMG_HEIGHT, IMG_WIDTH),
                             PillowResize::INTERPOLATION_BICUBIC);
    // cv::resize(image, resized_img, cv::Size(IMG_HEIGHT, IMG_WIDTH), 0, 0,
    //            cv::INTER_AREA);
    resized_img.convertTo(resized_img, CV_32FC1, 0.00392156862745098, 0);
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(resized_img, rgbChannels);
    for (int c = 0; c < 3; c++) {
      rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / 0.5,
                               (0.0 - 0.5) / 0.5);
    }

    // convert to array
    memcpy((void *)process_image.data(), (float *)rgbChannels[0].data,
           single_chn_size);
    memcpy((void *)(process_image.data() + image_area),
           (float *)rgbChannels[1].data, single_chn_size);
    memcpy((void *)(process_image.data() + image_area * 2),
           (float *)rgbChannels[2].data, single_chn_size);
    std::vector<fp16> feature;
    g_vila.vit_inference(process_image, feature);
    output_features.emplace_back(feature);
  }
}

void inferVLMModel(const std::string &prompt, const HalfMatrix &features,
                   VLMResult &result) {
  assert(g_vila.is_init == true);
  std::vector<int> tokens;
  g_vila.tokenizer_image_token(tokens, prompt);
  // infer tokens
  int token;
  float score;
  g_vila.forward_first(tokens, features, token, score);
  if (ONLY_FIRST_OUTPUT) {
    std::vector<int> pack = {29871, token};
    result.output_text = g_vila.tokenizer_decode(pack);
    result.confidence = score;
  } else {
    result.output_text.clear();
    result.confidence = score;
    while (token != g_vila.EOS) {
      std::vector<int> pack = {29871, token};
      result.output_text += g_vila.tokenizer_decode(pack);
      g_vila.forward_next(token, score);
    }
  }
}

void inferVLMModel(const std::string &prompt,
                   const std::vector<cv::Mat> &images, VLMResult &result) {
  HalfMatrix features;
  extractVLMVisualFeature(images, features);
  inferVLMModel(prompt, features, result);
}