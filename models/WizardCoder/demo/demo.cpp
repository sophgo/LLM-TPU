#include <bits/stdc++.h>
#include <bmlib_runtime.h>
#include <bmruntime_interface.h>
#include <getopt.h>
#include <cstdio>
#include "include/tokenizer.h"

static const int   NUM_LAYERS = 40;
static const int   MAX_LEN = 512;
static const float ATTENTION_MASK = -10000.;
static const int   num_heads = 48;

#define FYEL

inline long long get_elapsed(
        std::chrono::time_point<
                std::chrono::system_clock,
                std::chrono::duration<long, std::ratio<1, 1000000000>>>& last) {
    auto now = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(now - last)
            .count();
}

struct WizardCoder {
    WizardCoder() {}

    WizardCoder(const WizardCoder&) = delete;
    WizardCoder& operator=(const WizardCoder&) = delete;
    WizardCoder(WizardCoder&&) noexcept = default;
    WizardCoder& operator=(WizardCoder&&) noexcept = default;

    GPT2Tokenizer tokenizer;

    std::vector<bm_handle_t> handles;
    std::vector<int>         dev_ids;
    int                      num_device;
    bm_handle_t              handle;
    void*                    bmrt;

    struct WizardCoderEmbedding {
        bm_tensor_t input_ids_512, input_pos_512;
        bm_tensor_t input_ids_1, input_pos_1;
        bm_tensor_t hidden_states_512, hidden_states_1;
    } embedding;

    struct WizardCoderBlock {
        std::vector<bm_tensor_t> input_states;
        std::vector<bm_tensor_t> attention_mask;
        std::vector<bm_tensor_t> hidden_states;
        std::vector<bm_tensor_t> past_layers;
    };

    struct WizardCoderBlockCache {
        std::vector<bm_tensor_t> input_states;
        std::vector<bm_tensor_t> past_cache;
        std::vector<bm_tensor_t> attention_mask;
        std::vector<bm_tensor_t> hidden_states;
        std::vector<bm_tensor_t> current_cache;
    };

    std::vector<WizardCoderBlock>      blocks;
    std::vector<WizardCoderBlockCache> blocks_cache;

    struct WizardCoderLmHead {
        bm_tensor_t hidden_states;
        bm_tensor_t token;
    } lm_head;

    int token_length;

    std::unordered_map<std::string_view, const bm_net_info_t*> networks;

    void move2end(const bm_tensor_t& cache);
    int  forward_first(const std::vector<int>& token_ids);
    int  forward_next();
    void deinit();

    void stream_generate(const std::vector<int>& input_ids, int max_new_length);
    std::string generate(const std::vector<int>& input_ids, int max_new_length);

    std::string build_prompt(std::string_view) const;

    void init(std::string_view, const std::vector<int>&);

    void answer(std::string_view, int max_new_length = 500);

    void chat();
};

void WizardCoder::init(
        std::string_view        model_path,
        const std::vector<int>& devids) {
    auto tokenizer = GPT2Tokenizer::from_pretrained(VOCAB_DIR);
    if (!tokenizer) {
        std::cerr << "No tokenizer\n";
    }
    this->tokenizer = std::move(tokenizer.value());
    num_device = devids.size();
    blocks.resize(NUM_LAYERS);
    blocks_cache.resize(NUM_LAYERS);

    for (auto&& block : blocks) {
        block.attention_mask.resize(num_device);
        block.hidden_states.resize(num_device);
        block.input_states.resize(num_device);
        block.past_layers.resize(num_device);
    }

    for (auto&& block_cache : blocks_cache) {
        block_cache.current_cache.resize(num_device);
        block_cache.past_cache.resize(num_device);
        block_cache.attention_mask.resize(num_device);
        block_cache.hidden_states.resize(num_device);
        block_cache.input_states.resize(num_device);
    }

    for (auto id : devids) {
        bm_handle_t handle;
        if (bm_dev_request(&handle, id) != BM_SUCCESS) {
            std::cerr << "Error in bm_dev_request\n";
            return;
        }
        handles.push_back(handle);
    }

    handle = handles[0];
    auto& handle = handles[0];

    if (!(bmrt = bmrt_create_ex(&handle, num_device))) {
        std::cerr << "Error in bmrt_create_ex\n";
        return;
    }
    if (!bmrt_load_bmodel(bmrt, model_path.data())) {
        std::cerr << "Error in bmrt_load_bmodel\n";
        return;
    }

    const char** network_names{nullptr};
    bmrt_get_network_names(bmrt, &network_names);
    int num = bmrt_get_network_number(bmrt);
    for (int i = 0; i < num; i++) {
        networks[network_names[i]] =
                bmrt_get_network_info(bmrt, network_names[i]);
    }

    [&]() {
        bmrt_tensor(
                &embedding.input_ids_512,
                bmrt,
                networks["embedding"]->input_dtypes[0],
                networks["embedding"]->stages[1].input_shapes[0]);
        bmrt_tensor(
                &embedding.input_pos_512,
                bmrt,
                networks["embedding"]->input_dtypes[1],
                networks["embedding"]->stages[1].input_shapes[1]);
        bmrt_tensor(
                &embedding.hidden_states_512,
                bmrt,
                networks["embedding"]->output_dtypes[0],
                networks["embedding"]->stages[1].output_shapes[0]);
        bmrt_tensor(
                &embedding.input_ids_1,
                bmrt,
                networks["embedding"]->input_dtypes[0],
                networks["embedding"]->stages[0].input_shapes[0]);
        bmrt_tensor(
                &embedding.input_pos_1,
                bmrt,
                networks["embedding"]->input_dtypes[1],
                networks["embedding"]->stages[0].input_shapes[1]);
        bmrt_tensor(
                &embedding.hidden_states_1,
                bmrt,
                networks["embedding"]->output_dtypes[0],
                networks["embedding"]->stages[0].output_shapes[0]);
    }();

    [&]() {
        for (int i = 0; i < NUM_LAYERS; i++) {
            auto  name = std::string{"block_"} + std::to_string(i);
            auto  block_net = bmrt_get_network_info(bmrt, name.c_str());
            int   in_num = block_net->input_num / num_device;
            int   out_num = block_net->output_num / num_device;
            auto& block = blocks[i];

            for (int j = 0; j < num_device; j++) {
                bmrt_tensor_ex(
                        &block.input_states[j],
                        bmrt,
                        block_net->input_loc_devices[j * in_num + 0],
                        block_net->input_dtypes[j * in_num + 0],
                        block_net->stages[0].input_shapes[j * in_num + 0]);
                bmrt_tensor_ex(
                        &block.attention_mask[j],
                        bmrt,
                        block_net->input_loc_devices[j * in_num + 1],
                        block_net->input_dtypes[j * in_num + 1],
                        block_net->stages[0].input_shapes[j * in_num + 1]);
                bmrt_tensor_ex(
                        &block.hidden_states[j],
                        bmrt,
                        block_net->output_loc_devices[j * out_num + 0],
                        block_net->output_dtypes[j * out_num + 0],
                        block_net->stages[0].output_shapes[j * out_num + 0]);
                bmrt_tensor_ex(
                        &block.past_layers[j],
                        bmrt,
                        block_net->output_loc_devices[j * out_num + 1],
                        block_net->output_dtypes[j * out_num + 1],
                        block_net->stages[0].output_shapes[j * out_num + 1]);
            }
        }
    }();

    [&]() {
        for (int i = 0; i < NUM_LAYERS; i++) {
            auto  name = std::string{"block_cache_"} + std::to_string(i);
            auto  block_net = bmrt_get_network_info(bmrt, name.c_str());
            int   in_num = block_net->input_num / num_device;
            int   out_num = block_net->output_num / num_device;
            auto& block = blocks_cache[i];
            for (int j = 0; j < num_device; j++) {
                bmrt_tensor_ex(
                        &block.input_states[j],
                        bmrt,
                        block_net->input_loc_devices[j * in_num + 0],
                        block_net->input_dtypes[j * in_num + 0],
                        block_net->stages[0].input_shapes[j * in_num + 0]);

                bmrt_tensor_ex(
                        &block.past_cache[j],
                        bmrt,
                        block_net->input_loc_devices[j * in_num + 1],
                        block_net->input_dtypes[j * in_num + 1],
                        block_net->stages[0].input_shapes[j * in_num + 1]);

                bmrt_tensor_ex(
                        &block.attention_mask[j],
                        bmrt,
                        block_net->input_loc_devices[j * in_num + 2],
                        block_net->input_dtypes[j * in_num + 2],
                        block_net->stages[0].input_shapes[j * in_num + 2]);

                bmrt_tensor_ex(
                        &block.hidden_states[j],
                        bmrt,
                        block_net->output_loc_devices[j * out_num + 0],
                        block_net->output_dtypes[j * out_num + 0],
                        block_net->stages[0].output_shapes[j * out_num + 0]);

                bmrt_tensor_ex(
                        &block.current_cache[j],
                        bmrt,
                        block_net->output_loc_devices[j * out_num + 1],
                        block_net->output_dtypes[j * out_num + 1],
                        block_net->stages[0].output_shapes[j * out_num + 1]);
            }
        }
    }();

    [&]() {
        auto lm_head = bmrt_get_network_info(bmrt, "lm_head");
        bmrt_tensor(
                &this->lm_head.hidden_states,
                bmrt,
                lm_head->input_dtypes[0],
                lm_head->stages[0].input_shapes[0]);
        bmrt_tensor(
                &this->lm_head.token,
                bmrt,
                lm_head->output_dtypes[0],
                lm_head->stages[0].output_shapes[0]);
    }();
    return;
}

int WizardCoder::forward_first(const std::vector<int>& token_ids) {
    token_length = token_ids.size();
    auto attention_mask = std::make_unique<float[]>(MAX_LEN * MAX_LEN);
    auto position_id = std::make_unique<int[]>(MAX_LEN);
    for (int i = 0; i < MAX_LEN; i++) {
        for (int j = i + 1; j < MAX_LEN; j++)
            attention_mask[j + i * MAX_LEN] = -1000.0;
        if (i < token_length) position_id[i] = i;
    }

    std::vector<int>   one_input_nums{1};
    std::vector<int>   num_device_inputs_nums(num_device, 1);
    std::vector<void*> pos_id_data{position_id.get()};
    std::vector<void*> tok_id_data{(void*)token_ids.data()};

    std::vector<void*> attention_mask_data(
            num_device, (void*)attention_mask.get());

    bmrt_memcpy_s2d_parallel(
            bmrt,
            &embedding.input_ids_512,
            tok_id_data.data(),
            one_input_nums.data(),
            1);
    bmrt_memcpy_s2d_parallel(
            bmrt,
            &embedding.input_pos_512,
            pos_id_data.data(),
            one_input_nums.data(),
            1);

    bmrt_memcpy_s2d_parallel(
            bmrt,
            blocks.begin()->attention_mask.data(),
            attention_mask_data.data(),
            num_device_inputs_nums.data(),
            num_device);

    bm_tensor_t input_blocks[] = {
            embedding.input_ids_512, embedding.input_pos_512};

    bmrt_launch_tensor_ex(
            bmrt,
            "embedding",
            input_blocks,
            2,
            &embedding.hidden_states_512,
            1,
            true,
            false);

    bm_thread_sync(handle);

    std::vector<bm_tensor_t> inputs_block;
    std::vector<bm_tensor_t> outputs_block;

    for (int i = 0; i < num_device; i++) {
        embedding.hidden_states_512.shape = blocks[0].input_states[0].shape;
        inputs_block.push_back(embedding.hidden_states_512);
        inputs_block.push_back(blocks[0].attention_mask[i]);
        outputs_block.push_back(embedding.hidden_states_512);
        outputs_block.push_back(blocks[0].past_layers[i]);
    }

    for (int i = 0; i < NUM_LAYERS; i++) {
        auto name = std::string{"block_"} + std::to_string(i);
        for (int j = 0; j < num_device; j++) {
            outputs_block[1] = blocks[i].past_layers[j];
        }

        bmrt_launch_tensor_ex(
                bmrt,
                name.c_str(),
                inputs_block.data(),
                inputs_block.size(),
                outputs_block.data(),
                outputs_block.size(),
                true,
                false);

        bm_thread_sync(handle);

        for (int j = 0; j < num_device; j++) {
            move2end(blocks[i].past_layers[j]);
        }

        bm_thread_sync(handle);
    }

    auto bytes =
            bm_mem_get_device_size(embedding.hidden_states_512.device_mem) /
            MAX_LEN;

    bm_memcpy_d2d_byte(
            handle,
            lm_head.hidden_states.device_mem,
            0,
            embedding.hidden_states_512.device_mem,
            (token_length - 1) * bytes,
            bytes);

    bmrt_launch_tensor_ex(
            bmrt,
            "lm_head",
            &lm_head.hidden_states,
            1,
            &lm_head.token,
            1,
            true,
            false);
    int token = 0;
    bm_memcpy_d2s(handle, &token, lm_head.token.device_mem);

    ++token_length;
    return token;
}

void WizardCoder::move2end(const bm_tensor_t& cache) {
    auto sz = bm_mem_get_device_size(cache.device_mem);
    auto bytes = sz / MAX_LEN;
    auto len = token_length * bytes;
    bm_memcpy_d2d(handle, cache.device_mem, sz - len, cache.device_mem, 0, len);
}

int WizardCoder::forward_next() {
    int                pid = token_length - 1;
    std::vector<void*> input_pid_data{&pid};
    std::vector<int>   embedding_inputs_num{1};
    bmrt_memcpy_s2d_parallel(
            bmrt,
            &embedding.input_pos_1,
            input_pid_data.data(),
            embedding_inputs_num.data(),
            1);

    bmrt_tensor_with_device(
            &embedding.input_ids_1,
            lm_head.token.device_mem,
            embedding.input_ids_1.dtype,
            embedding.input_ids_1.shape);

    bm_tensor_t input_blocks[] = {embedding.input_ids_1, embedding.input_pos_1};
    bmrt_launch_tensor_ex(
            bmrt,
            "embedding",
            input_blocks,
            2,
            &embedding.hidden_states_1,
            1,
            true,
            false);

    bm_thread_sync(handle);

    auto attention_mask = std::make_unique<float[]>(1 + MAX_LEN);
    for (int i = 0; i < MAX_LEN - token_length + 1; i++)
        attention_mask[i] = -1000;

    std::vector<int>   input_nums(num_device, 1);
    std::vector<void*> attention_mask_data(num_device, attention_mask.get());

    bmrt_memcpy_s2d_parallel(
            bmrt,
            blocks_cache.begin()->attention_mask.data(),
            attention_mask_data.data(),
            input_nums.data(),
            num_device);

    std::vector<bm_tensor_t> inputs_block;
    std::vector<bm_tensor_t> outputs_block;

    for (int i = 0; i < num_device; i++) {
        inputs_block.push_back(embedding.hidden_states_1);
        inputs_block.push_back(blocks[0].past_layers[i]);
        inputs_block.push_back(blocks_cache[0].attention_mask[i]);
        outputs_block.push_back(embedding.hidden_states_1);
        outputs_block.push_back(blocks_cache[0].current_cache[i]);
    }

    for (int i = 0; i < NUM_LAYERS; i++) {
        auto name = std::string{"block_cache_"} + std::to_string(i);

        for (int j = 0; j < num_device; j++) {
            inputs_block[1] = blocks[i].past_layers[j];
            outputs_block[1] = blocks_cache[i].current_cache[j];
        }

        bmrt_launch_tensor_ex(
                bmrt,
                name.c_str(),
                inputs_block.data(),
                inputs_block.size(),
                outputs_block.data(),
                outputs_block.size(),
                true,
                false);

        bm_thread_sync(handle);

        auto totalsize = bm_mem_get_device_size(
                                 blocks_cache[0].current_cache[0].device_mem) /
                513;
        for (int j = 0; j < num_device; j++) {
            bm_memcpy_d2d(
                    handle,
                    blocks[i].past_layers[j].device_mem,
                    0,
                    blocks_cache[i].current_cache[j].device_mem,
                    totalsize,
                    totalsize * 512);
        }
    }

    bmrt_launch_tensor_ex(
            bmrt,
            "lm_head",
            &embedding.hidden_states_1,
            1,
            &lm_head.token,
            1,
            true,
            false);
    bm_thread_sync(handle);

    int token = 0;
    ++token_length;
    bm_memcpy_d2s(handle, &token, lm_head.token.device_mem);

    return token;
}

std::string WizardCoder::build_prompt(std::string_view input_str) const {
    return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
           "### Instruction:\n" +
            std::string{input_str} + "\n\n### Response:";
}

void WizardCoder::stream_generate(
        const std::vector<int>& input_ids,
        int                     max_new_length) {
    int cnt = 1;

    auto const input_token_len = input_ids.size();
    auto       start_time = std::chrono::high_resolution_clock::now();
    auto       token = forward_first(input_ids);

    auto FTL = get_elapsed(start_time);

    start_time = std::chrono::high_resolution_clock::now();

    while (++cnt < max_new_length && cnt + input_token_len <= MAX_LEN) {
        auto result = tokenizer.decode_id(token, true);
        if (result == "<|endoftext|>") break;
        std::cout << result << std::flush;
        token = forward_next();
    }

    auto total = get_elapsed(start_time);

    std::cout << FYEL("\n\nInference Time: ") << (total + FTL)
              << FYEL(" ms\nToken: ") << cnt << FYEL(" FTL: ") << FTL
              << FYEL(" ms\nRate: ") << (cnt - 1) * 1000.0 / total
              << FYEL(" Token/Sec\n");
}

void WizardCoder::answer(std::string_view input_str, int max_new_length) {
    auto prompt = build_prompt(input_str);
    auto input_ids = tokenizer.encode(prompt);
    stream_generate(input_ids, max_new_length);
}

void WizardCoder::chat() {
    while (true) {
        std::cout << "\nQuestion: ";
        std::string input_str;
        std::getline(std::cin, input_str);
        if (input_str == "exit") {
            break;
        }

        std::cout << "\nAnswer: " << std::flush;
        answer(input_str);
        std::cout << std::endl;
    }
}

static void split(
        const std::string&        s,
        const std::string&        delim,
        std::vector<std::string>& ret) {
    size_t last = 0;
    size_t index = s.find_first_of(delim, last);
    while (index != std::string::npos) {
        ret.push_back(s.substr(last, index - last));
        last = index + 1;
        index = s.find_first_of(delim, last);
    }
    if (last < s.length()) {
        ret.push_back(s.substr(last));
    }
}

static std::vector<int> parseCascadeDevices(const std::string& str) {
    std::vector<int>         devices;
    std::vector<std::string> sub_str;
    split(str, ",", sub_str);
    for (auto& s : sub_str) {
        devices.push_back(std::atoi(s.c_str()));
    }
    return devices;
}

void processArguments(
        int               argc,
        char*             argv[],
        std::string&      llama_model,
        std::vector<int>& devices) {
    struct option longOptions[] = {
            {"model", required_argument, nullptr, 'm'},
            {"dev_id", required_argument, nullptr, 'd'},
            {nullptr, 0, nullptr, 0}};

    int optionIndex = 0;
    int option;

    while ((option = getopt_long(
                    argc, argv, "m:d:", longOptions, &optionIndex)) != -1) {
        switch (option) {
            case 'm':
                llama_model = optarg;
                break;
            case 'd':
                devices = parseCascadeDevices(optarg);
                break;
            case '?':
                exit(EXIT_FAILURE);
            default:
                exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv) {
    printf("Demo for Wizardcoder-15B in BM1684X\n");
    printf("The location of vocab.json is: %s\n", VOCAB_DIR);

    std::string      wizardcoder_model = "wizardcoder-15b_int4_1dev.bmodel";
    std::vector<int> devices = {0};

    processArguments(argc, argv, wizardcoder_model, devices);

    printf("Init Environment ...\n");
    WizardCoder model;
    model.init(wizardcoder_model, devices);

    model.chat();

    return 0;
}
