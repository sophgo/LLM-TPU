#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sophon.sail as sail
import argparse
import time
from transformers import AutoTokenizer
import numpy as np

class DeepSeek:
    def __init__(self, args):
        self.input_str = ""
        system_prompt = None
        self.system = {"role":"system","content":system_prompt}
        if system_prompt is not None:
            self.history = [self.system]
        else:
            self.history = []

        # load tokenizer
        print("Load " + args.token + " ...")
        self.sp = AutoTokenizer.from_pretrained(args.token, trust_remote_code=True)

        # warm up
        self.sp.decode([0]) 
        self.EOS = self.sp.eos_token_id
        print("Done!")

        # load bmodel
        # 这里devio，后面都没有创建系统内存的tensor
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.DEVIO)
        self.handle = sail.Handle(args.dev_id)
        self.graph_name = self.net.get_graph_names()
        self.NUM_LAYERS, self.MAX_LEN, self.HIDDEN_SIZE = self.auto_parameters("block_0")
        
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]

        # name, input_idx, shape, 
        # tensors:
        # forward_first: embedding_tensor
        self.first_embed_input = self.init_input_tensor(self.name_embed, 0, [1, self.MAX_LEN])
        self.first_embed_output = self.init_input_tensor(self.name_embed, 0, [1, self.MAX_LEN, self.HIDDEN_SIZE], False)

        # forward_next: embedding_tensor
        self.next_embed_input = self.init_input_tensor(self.name_embed_cache, 0, [1, 1])
        self.next_embed_output = self.init_input_tensor(self.name_embed_cache, 0, [1,  self.HIDDEN_SIZE], False)

        # forward_first: hidden_state
        self.first_hidden_input = self.init_input_tensor(self.name_blocks[0], 0)
        self.first_hidden_output = self.init_input_tensor(self.name_blocks[0], 0, None, False)

        # forward_next: hidden_state
        self.next_hidden_input = self.init_input_tensor(self.name_blocks_cache[0], 0)
        self.next_hidden_output = self.init_input_tensor(self.name_blocks_cache[0], 0, None, False)

        # forward_first: position_id_tensor 和 attention_mask_tensor
        self.first_pid = self.init_input_tensor(self.name_blocks[0], 1)
        self.first_attention = self.init_input_tensor(self.name_blocks[0], 2)
       
        # forward_next: position_id_tensor and attention_mask_tensor
        self.next_pid = self.init_input_tensor(self.name_blocks_cache[0], 1)
        self.next_attention = self.init_input_tensor(self.name_blocks_cache[0], 2)

        # forward_next: present_key / present_value (for update kv_cache)
        self.present_key = self.init_input_tensor(self.name_blocks_cache[0], 1, None, False)
        self.present_value = self.init_input_tensor(self.name_blocks_cache[0], 2, None, False)

        # forward_first: key_tensor 和 value_tensor
        self.past_key_output = []
        self.past_value_output = []

        # forward_next: cache block的kv tensor名
        self.cache_key_input = []
        self.cache_key_output = []
        self.cache_value_input = []
        self.cache_value_output = []

        for _ in range(self.NUM_LAYERS):
            
            self.past_key_output.append(self.init_input_tensor(self.name_blocks[0], 1, None, False))
            self.past_value_output.append(self.init_input_tensor(self.name_blocks[0], 2, None, False))

            self.cache_key_input.append(self.init_input_tensor(self.name_blocks_cache[0], 3))
            self.cache_key_output.append(self.init_input_tensor(self.name_blocks_cache[0], 1, None, False))

            self.cache_value_input.append(self.init_input_tensor(self.name_blocks_cache[0], 4))
            self.cache_value_output.append(self.init_input_tensor(self.name_blocks_cache[0], 2, None, False))

        # lm_head tensor
        self.lm_input = self.init_input_tensor(self.name_lm, 0)
        self.lm_output = self.init_input_tensor(self.name_lm, 0, None, False)

        self.token_length = 0
        self.round = 0

    def init_input_tensor(self, name, input_idx, shape=None, input_type=True):
        tensor = {}
        if input_type:
            tensor["name"] = self.net.get_input_names(name)[input_idx]
            tensor["shape"] = self.net.get_input_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_input_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True)
        else:
            tensor["name"] = self.net.get_output_names(name)[input_idx]
            tensor["shape"] = self.net.get_output_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_output_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True) 
        return tensor

    def auto_parameters(self, block_name):
        NUM_LAYERS = (len(self.net.get_graph_names()) - 2) // 2
        _, MAX_LEN, HIDDEN_SIZE = self.first_hidden_input_shape = self.net.get_input_shape(block_name, self.net.get_input_names(block_name)[0])
        return NUM_LAYERS, MAX_LEN, HIDDEN_SIZE
    
    def _make_context(self, input_str):
        messages = self.history + [{"role":"user", "content":input_str}]
        input_ids = self.sp.apply_chat_template(messages, add_generation_prompt=True, return_tensors="np")[0].tolist()
        return input_ids

    def forward_first(self, token):
        # Keep
        input_ids = np.zeros(self.MAX_LEN, np.int32)
        position_id = np.zeros(self.MAX_LEN, np.int32)
        attention_mask = np.ones(self.MAX_LEN*self.MAX_LEN, np.float32) * (-10000.0)
        input_ids[:len(token)] = token
        self.token_length = len(token)
        for i in range(self.token_length):
            position_id[i] = i
        for i in range(self.token_length):
            for j in range(self.MAX_LEN):
                if (j <= i):
                    attention_mask[i*self.MAX_LEN + j] = 0
        # import pdb;pdb.set_trace()
        # embedding
        input_ids = input_ids.reshape(1, -1)
        self.first_embed_input["data"].update_data(input_ids)
        input_embed_tensors = {self.first_embed_input["name"]: self.first_embed_input["data"]}
        output_embed_tensors = {self.first_embed_output["name"]: self.first_embed_output["data"]}
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)

        # blocks
        self.first_hidden_tensor = self.first_embed_output["data"]
        self.first_hidden_tensor.reshape(self.first_hidden_input["shape"])
        self.first_pid["data"].update_data(position_id.reshape(self.first_pid["shape"]))
        self.first_attention["data"].update_data(attention_mask.reshape(self.first_attention["shape"]).astype(np.uint16))

        input_blocks_tensors = {self.first_hidden_input["name"]: self.first_hidden_tensor, 
                                self.first_pid["name"]: self.first_pid["data"], 
                                self.first_attention["name"]: self.first_attention["data"]}

        for i in range(self.NUM_LAYERS):        

            output_blocks_tensors = {self.first_hidden_output["name"]: self.first_hidden_tensor,
                                    self.past_key_output[i]["name"]: self.past_key_output[i]["data"],
                                    self.past_value_output[i]["name"]: self.past_value_output[i]["data"]}
            
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)
        
        # lm_head
        # hidden_states 的最后一个位置的元素取出来作为 lm_head的输入
        copy_len = self.first_hidden_tensor.shape()[-1]
        self.lm_input["data"].sync_d2d(self.first_hidden_tensor,
                                      (self.token_length-1)* copy_len,  
                                      0, 
                                      copy_len)
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input["data"]}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}

        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        # print("First Token is : ", int(self.lm_output["data"].asnumpy()))
        return int(self.lm_output["data"].asnumpy())

    def forward_next(self, ):
        attention_mask = np.zeros(self.MAX_LEN+1, np.float32)
        for i in range(self.token_length-1, self.MAX_LEN):
            attention_mask[i] = -10000.0

        position_id = np.array(self.token_length - 1, dtype=np.int32)
        # embedding
        self.next_embed_input["data"] = self.lm_output["data"]
        self.next_embed_input["data"].reshape(self.next_embed_input["shape"])
        # import pdb;pdb.set_trace()
        input_embed_tensors = {self.next_embed_input["name"]: self.next_embed_input["data"]}
        output_embed_tensors = {self.next_embed_output["name"]: self.next_embed_output["data"]}
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)

        # blocks
        self.next_pid["data"].update_data(position_id.reshape(self.next_pid["shape"]))
        self.next_attention["data"].update_data(attention_mask.reshape(self.next_attention["shape"]).astype(np.uint16))

        self.next_hidden_tensor = self.next_embed_output["data"]
        self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {self.next_hidden_input["name"]: self.next_hidden_tensor, 
                                        self.next_pid["name"]: self.next_pid["data"], 
                                        self.next_attention["name"]: self.next_attention["data"], 
                                        self.cache_key_input[i]["name"]: self.past_key_output[i]["data"], 
                                        self.cache_value_input[i]["name"]: self.past_value_output[i]["data"]}
            outputs_block_cache_tensors = {self.next_hidden_output["name"]: self.next_hidden_tensor,
                                        self.cache_key_output[i]["name"]: self.present_key["data"],
                                        self.cache_value_output[i]["name"]: self.present_value["data"]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

            # update kv_cache()
            unit_size = self.present_key["shape"][-1]*self.present_key["shape"][-2]
            self.past_key_output[i]["data"].sync_d2d(self.present_key["data"], 0, (self.token_length-1)*unit_size, unit_size)
            self.past_value_output[i]["data"].sync_d2d(self.present_value["data"], 0, (self.token_length-1)*unit_size, unit_size)

        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input["shape"])
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input_tensor}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return int(self.lm_output["data"].asnumpy())
        
    def chat(self):
        while True:
            self.input_str = input("\nQuestion: ")
            if self.input_str == "exit":
                break
            # import pdb;pdb.set_trace()
            token = self._make_context(self.input_str)
            print("\nAnswer: ")
            self.answer(token)
            print()

    def answer(self, tokens):
        tok_num = 0
        self.answer_cur = ""

        # input is empty
        if tokens is None or len(tokens) == 0:
            print("Sorry: your question is too wierd!!")
            return
        if self.token_length > self.MAX_LEN:
            print("The maximum question length should be shorter than {} but we get {} instead.".format(self.MAX_LEN, self.token_length))
            return

        first_start = time.time()
        token = self.forward_first(tokens)
        first_end = time.time()
        pre_token = 0
        while token != self.EOS and self.token_length < self.MAX_LEN:
            pre_ids = [pre_token]
            ids = [pre_token, token]
            pre_word = self.sp.decode(pre_ids)
            word = self.sp.decode(ids)
            diff = word[len(pre_word):]
            self.answer_cur += diff
            print(diff, flush=True, end='')
            if self.token_length < self.MAX_LEN:
                self.token_length += 1
            tok_num += 1
            token = self.forward_next()

        # 计时
        next_end = time.time()
        first_duration = first_end-first_start
        next_duration = next_end-first_end
        tps = tok_num / next_duration

        if self.token_length >= self.MAX_LEN - 128:
            print("... (reach the maximal length)", flush=True, end='')
            self.history += [{"role":"user", "content":self.input_str},
                                 {"role":"assistant", "content":self.answer_cur}]
            self.history = [self.system] + self.history[(len(self.history)-1)//2 + 1:]
        else:
            self.history += [{"role":"user", "content":self.input_str},
                                 {"role":"assistant", "content":self.answer_cur}]

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

def main(args):
    deepseek = DeepSeek(args)
    deepseek.chat()
    pass

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default='./model/deepseek-6.7b_int4_1dev.bmodel', help='path of bmodel')
    parser.add_argument('--token', type=str, default='./token_config/', help='path of tokenizer')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done')

# 10994 727
