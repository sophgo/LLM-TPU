import os
import sys
import math
import copy
import glob
import json
import time
import base64
import logging
import warnings
import argparse
import functools
import subprocess
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Tuple

from yaspin import yaspin

import onnx
import torch
import numpy as np
from onnx import numpy_helper
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

GREEN_COLOR = "\033[92m"  # ANSI escape code for green text
RESET_COLOR = "\033[0m"

def logging(message):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(message)  # 打印传入的消息
            return func(*args, **kwargs)  # 调用原函数
        return wrapper
    return decorator

class ModelMapper:
    def __init__(self):
        self.attrs = []
        self.mapper = dict()
        self.regist_models()

    def get_map(self, config):
        model_type = config.model_type
        if model_type == 'chatglm':
            if hasattr(config, 'vocab_size') and config.vocab_size == 130528:
                model_type = 'chatglm'
            else:
                model_type = 'chatglm2'
        if model_type in self.mapper:
            return model_type, self.mapper[model_type]
        return model_type, self.default_map

    def regist(self, model_type, model_map):
        assert('config' in model_map and
               'decoder' in model_map and
               'attention' in model_map)
        self.mapper[model_type] = model_map

    def regist_models(self):
        self.defualt_map()
        llama_map = self.default_map
        self.regist('llama', llama_map)
        self.regist('qwen2', llama_map)
        self.regist('internlm', llama_map)
        baichuan_map = copy.deepcopy(self.default_map)
        baichuan_map[self.attention_key] = {
            'qkv_proj': 'W_pack',
            'o_proj': 'o_proj'
        }
        self.regist('baichuan', baichuan_map)
        self.regist_qwen()
        self.regist_glm()
        self.regist_glm2()
        self.regist_phi()

    def regist_qwen(self):
        qwen_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_hidden_layers',
                'rope_theta': 'rotary_emb_base',
            },
            'model': {
                'lm_': 'lm_head',
                'embed_': 'transformer.wte',
                'blocks_': 'transformer.h',
                'final_layernorm_': 'transformer.ln_f',
                'visual': 'transformer.visual'
            },
            'decoder': {
                'self_attn': 'attn',
                'mlp': 'mlp',
                'input_layernorm': 'ln_1',
                'post_attention_layernorm': 'ln_2'
            },
            'attention': {
                'qkv_proj': 'c_attn',
                'o_proj': 'c_proj'
            }
        }
        self.regist('qwen', qwen_map)

    def regist_glm(self):
        glm_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_layers'
            },
            'model': {
                'lm_': 'lm_head',
                'embed_': 'transformer.word_embeddings',
                'blocks_': 'transformer.layers',
                'final_layernorm_': 'transformer.final_layernorm',
            },
            'decoder': {
                'self_attn': 'attention',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm'
            },
            'attention': {
                'qkv_proj': 'query_key_value',
                'o_proj': 'dense'
            }
        }
        self.regist('chatglm', glm_map)

    def regist_glm2(self):
        glm2_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_key_value_heads': 'multi_query_group_num',
                'num_hidden_layers': 'num_layers',
            },
            'model': {
                'lm_': 'transformer.output_layer',
                'embed_': 'transformer.embedding.word_embeddings',
                'blocks_': 'transformer.encoder.layers',
                'final_layernorm_': 'transformer.encoder.final_layernorm',
            },
            'decoder': {
                'self_attn': 'self_attention',
                'mlp': 'mlp',
                'input_layernorm': 'input_layernorm',
                'post_attention_layernorm': 'post_attention_layernorm'
            },
            'attention': {
                'qkv_proj': 'query_key_value',
                'o_proj': 'dense'
            }
        }
        self.regist('chatglm2', glm2_map)

    def regist_phi(self):
        phi_map = {
            'config': {
                'hidden_size': 'n_embd',
                'num_attention_heads': 'n_head',
                'num_hidden_layers': 'n_layer',
                'rotary_dim': 'rotary_dim'
            },
            'model': {
                'lm_': 'lm_head.linear',
                'embed_': 'transformer.embd.wte',
                'blocks_': 'transformer.h',
                'final_layernorm_': 'lm_head.ln',
            },
            'decoder': {
                'self_attn': 'mixer',
                'mlp': 'mlp',
                'input_layernorm': 'ln',
            },
            'attention': {
                'qkv_proj': 'Wqkv',
                'o_proj': 'out_proj'
            }
        }
        self.regist('phi-msft', phi_map)

    def defualt_map(self):
        # default map is `LlamaForCausalLM`
        self.config_key = 'config'
        self.model_key = 'model'
        self.decoder_key = 'decoder'
        self.attention_key = 'attention'
        self.default_config = {
            'hidden_size': 'hidden_size',
            'num_attention_heads': 'num_attention_heads',
            'num_hidden_layers': 'num_hidden_layers',
            'num_key_value_heads': 'num_key_value_heads',
            'rope_theta': 'rope_theta'
        }
        self.defualt_model = {
            'lm_': 'lm_head',
            'embed_': 'model.embed_tokens',
            'blocks_': 'model.layers',
            'final_layernorm_': 'model.norm',
        }
        self.default_decoder = {
            'self_attn': 'self_attn',
            'mlp': 'mlp',
            'input_layernorm': 'input_layernorm',
            'post_attention_layernorm': 'post_attention_layernorm'
        }
        self.default_attention = {
            'q_proj': 'q_proj',
            'k_proj': 'k_proj',
            'v_proj': 'v_proj',
            'o_proj': 'o_proj'
        }
        self.default_map = {
            'config': self.default_config,
            'model': self.defualt_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }

    @staticmethod
    def do_map(dst, src, map):
        for dst_attr, src_attr in map.items():
            attributes = src_attr.split('.')
            obj = src
            for attr in attributes:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    obj = None
                    break
            setattr(dst, dst_attr, obj)

class OnnxRebuild:
    def __init__(self):
        self.onnx_model = None

    def rebuild_weights(self, ref_path, torch_model, save_path):
        """
        Rebuild the weights from the PyTorch model into the ONNX model's initializers.
        Saves the updated ONNX model to the specified output path.
        """
        if self.onnx_model is None:
            self.onnx_model = onnx.load(ref_path)
        self.state_dict = torch_model.state_dict()

        # Map of name to initializer in ONNX model
        initializer_map = {init.name: init for init in self.onnx_model.graph.initializer}
        # Map of name to parameter data in PyTorch model
        pytorch_params = {name: param.detach().cpu().numpy() for name, param in self.state_dict.items()}

        # Try to match and update the weights
        matched = 0
        for name, param in pytorch_params.items():
            # Try to find a matching initializer in the ONNX model
            # Possible name adjustments if needed
            search_names = [name, name.replace('module.', '')]
            for init_name in search_names:
                if init_name in initializer_map:
                    init = initializer_map[init_name]
                    # Create a new TensorProto from the numpy array
                    new_init = numpy_helper.from_array(param, init_name)
                    # Replace the old initializer with the new one
                    self._replace_initializer(init, new_init)
                    matched += 1
                    break
        
        if matched != len(initializer_map) or matched != len(pytorch_params):
            raise ValueError("Match failed in onnx rebuild")

        # Save the modified model
        onnx.save(self.onnx_model, save_path)

    def _replace_initializer(self, old_init, new_init):
        """
        Replaces the contents of an existing initializer with new data.

        Args:
            old_init (onnx.onnx_ml_pb2.TensorProto): The existing initializer to be replaced.
            new_init (onnx.onnx_ml_pb2.TensorProto): The new initializer with updated data.
        """
        old_init.CopyFrom(new_init)

class BmodelConverter:
    def __init__(self, dst_path, config, args):
        self.dst_path = dst_path
        self.relative_onnx_path = "../onnx"
        self.bmodel_path = os.path.join(self.dst_path, "bmodel")
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        self.chip = args.chip
        self.quantize = args.quantize
        self.max_workers = args.max_workers
        self.num_device = args.num_device
        self.tpu_mlir_path = args.tpu_mlir_path
        self.seq_length = args.seq_length
        if args.bmodel:
            self.out_model = args.bmodel
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.out_model = f"{config.model_type}_{self.quantize}_seq{self.seq_length}_{timestamp}.bmodel"

        self.env = self._get_environment()

    def _get_environment(self):
        """
        Sources the envsetup.sh script and captures the environment variables.
        Returns a dictionary of the updated environment.
        """
        command = ["bash", "-c", "source envsetup.sh && env"]
        try:
            # Run the command in the tpu_mlir_path directory
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.tpu_mlir_path, text=True
            )
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Error sourcing envsetup.sh: {stderr}")

            # Parse the environment variables
            env = os.environ.copy()
            for line in stdout.splitlines():
                key, _, value = line.partition("=")
                env[key] = value
            return env
        except Exception as e:
            raise RuntimeError(f"Failed to get environment: {e}")

    def run_command(self, command, env):
        print(f"{GREEN_COLOR}Executing command: \n{' '.join(command)}{RESET_COLOR}")  # Print the command in green
        subprocess.run(command, check=True, env=env)

    def compile_lm_head(self, env, quantize):
        name = "lm_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
        else:
            path = os.path.join(self.relative_onnx_path, f"{name}.pt")
            transform_args = [
                'model_transform.py',
                f'--model_name {name}',
                f'--model_def {path}',
                f'--input_shapes [[1,{self.hidden_size}]]',
                f'--mlir {name}.mlir'
            ]
            deploy_args = [
                'model_deploy.py',
                f'--mlir {name}.mlir',
                f'--quantize {quantize}',
                '--quant_input',
                f'--chip {self.chip}',
                f'--num_device {self.num_device}',
                f'--model {name}.bmodel'
            ]
            self.run_command(['bash', '-c', ' '.join(transform_args)], env)
            self.run_command(['bash', '-c', ' '.join(deploy_args)], env)

    def compile_greedy_head(self, env):
        name = "greedy_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
        else:
            path = os.path.join(self.relative_onnx_path, f"{name}.onnx")
            transform_args = [
                'model_transform.py',
                f'--model_name {name}',
                f'--model_def {path}',
                f'--mlir {name}.mlir'
            ]
            deploy_args = [
                'model_deploy.py',
                f'--mlir {name}.mlir',
                f'--chip {self.chip}',
                f'--model {name}.bmodel'
            ]
            self.run_command(['bash', '-c', ' '.join(transform_args)], env)
            self.run_command(['bash', '-c', ' '.join(deploy_args)], env)

    def compile_penalty_head(self, env):
        name = "penalty_sample_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
        else:
            path = os.path.join(self.relative_onnx_path, f"{name}.onnx")
            transform_args = [
                'model_transform.py',
                f'--model_name {name}',
                f'--model_def {path}',
                f'--mlir {name}.mlir'
            ]
            deploy_args = [
                'model_deploy.py',
                f'--mlir {name}.mlir',
                f'--chip {self.chip}',
                f'--model {name}.bmodel'
            ]
            self.run_command(['bash', '-c', ' '.join(transform_args)], env)
            self.run_command(['bash', '-c', ' '.join(deploy_args)], env)

    def compile_block(self, i, env, quantize):
        name = f"block_{i}"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
        else:
            path = os.path.join(self.relative_onnx_path, f"{name}.onnx")
            transform_args = [
                'model_transform.py',
                f'--model_name {name}',
                f'--model_def {path}',
                f'--mlir {name}.mlir'
            ]
            deploy_args = [
                'model_deploy.py',
                f'--mlir {name}.mlir',
                f'--quantize {quantize}',
                '--quant_input',
                '--quant_output',
                f'--chip {self.chip}',
                f'--num_device {self.num_device}',
                f'--model {name}.bmodel'
            ]
            self.run_command(['bash', '-c', ' '.join(transform_args)], env)
            self.run_command(['bash', '-c', ' '.join(deploy_args)], env)

    def compile_block_cache(self, i, env, quantize):
        name = f"block_cache_{i}"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
        else:
            path = os.path.join(self.relative_onnx_path, f"{name}.onnx")
            transform_args = [
                'model_transform.py',
                f'--model_name {name}',
                f'--model_def {path}',
                f'--mlir {name}.mlir'
            ]
            deploy_args = [
                'model_deploy.py',
                f'--mlir {name}.mlir',
                f'--quantize {quantize}',
                '--quant_input',
                '--quant_output',
                f'--chip {self.chip}',
                '--addr_mode=io_alone',
                f'--num_device {self.num_device}',
                f'--model {name}.bmodel'
            ]
            self.run_command(['bash', '-c', ' '.join(transform_args)], env)
            self.run_command(['bash', '-c', ' '.join(deploy_args)], env)

    def combine(self, env):
        bmodel_list = []
        for i in range(self.num_layers):
            bmodel_list = bmodel_list + [f"block_{i}.bmodel", f"block_cache_{i}.bmodel"]
        bmodel_list += ["lm_head.bmodel", "greedy_head.bmodel", "penalty_sample_head.bmodel"]

        bmodel_list = [os.path.join(self.bmodel_path, b) for b in bmodel_list]

        combine_args = [
            'model_tool',
            '--combine',
            ' '.join(bmodel_list),
            '-o',
            os.path.join(self.dst_path, self.out_model)
        ]
        self.run_command(['bash', '-c', ' '.join(combine_args)], env)

    def compile(self):
        # Create the bmodel directory if it doesn't exist
        os.makedirs(self.bmodel_path, exist_ok=True)

        quantize = self.quantize
        half_precision_quantize = "f16" if "f16" in self.quantize else "bf16"

        # Compile heads
        ori_path = os.getcwd()
        os.chdir(self.bmodel_path)

        self.compile_lm_head(self.env, quantize)
        self.compile_greedy_head(self.env)
        self.compile_penalty_head(self.env)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(self.num_layers):
                futures.append(executor.submit(self.compile_block, i, self.env, quantize))
                futures.append(executor.submit(self.compile_block_cache, i, self.env, quantize))
            # Wait for all threads to complete
            for future in concurrent.futures.as_completed(futures):
                # This will raise exceptions if any occurred during thread execution
                future.result()

        # Optionally, change back to the original directory
        os.chdir(ori_path)

        # compile all bmodel
        self.combine(self.env)

# some wrapper class for export
class Embedding(torch.nn.Module):
    def __init__(self, embed, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed = embed

    def forward(self, input_ids):
        return self.embed(input_ids).view(-1, 1, self.hidden_size)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class Attention(torch.nn.Module):
    def __init__(self, attn, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rotary = config.rotary
        ModelMapper.do_map(self, attn, config.model_map['attention'])
        if hasattr(self, 'qkv_proj') and self.qkv_proj is not None:
            # split qkv linear to q, k, v
            split_sizes = [self.hidden_size] * 3
            if self.qkv_proj.weight.shape[0] != self.hidden_size * 3:
                # M/GQA
                qkv_hidden_size = self.qkv_proj.weight.shape[0]
                kv_hidden_size = (qkv_hidden_size - self.hidden_size) // 2
                split_sizes = [self.hidden_size, kv_hidden_size, kv_hidden_size]
            self.q_proj = torch.nn.Linear(self.hidden_size, split_sizes[0])
            self.k_proj = torch.nn.Linear(self.hidden_size, split_sizes[1])
            self.v_proj = torch.nn.Linear(self.hidden_size, split_sizes[2])
            if config.model_type == 'chatglm':
                # chatglm-6b
                qkv_weight = self.qkv_proj.weight.data.view(self.num_heads, 3, self.head_dim, self.hidden_size)
                self.q_proj.weight.data = qkv_weight[:, 0, :, :].reshape(self.hidden_size, self.hidden_size)
                self.k_proj.weight.data = qkv_weight[:, 1, :, :].reshape(self.hidden_size, self.hidden_size)
                self.v_proj.weight.data = qkv_weight[:, 2, :, :].reshape(self.hidden_size, self.hidden_size)
                qkv_bias = self.qkv_proj.bias.data.view(self.num_heads, 3, self.head_dim)
                self.q_proj.bias.data = qkv_bias[:, 0, :].reshape(self.hidden_size)
                self.k_proj.bias.data = qkv_bias[:, 1, :].reshape(self.hidden_size)
                self.v_proj.bias.data = qkv_bias[:, 2, :].reshape(self.hidden_size)
            else:
                # other
                qw, kw, vw = torch.split(self.qkv_proj.weight, split_sizes)
                self.q_proj.weight.data = qw
                self.k_proj.weight.data = kw
                self.v_proj.weight.data = vw
                if self.qkv_proj.bias is not None:
                    qb, kb, vb = torch.split(self.qkv_proj.bias, split_sizes)
                    self.q_proj.bias.data = qb
                    self.k_proj.bias.data = kb
                    self.v_proj.bias.data = vb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        kv_seq_len = key_states.shape[1]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[1]

        # rope
        cos, sin = self.rotary.cos[position_ids], self.rotary.sin[position_ids]
        query_states = self.rotary.apply_rotary_pos(query_states, cos, sin)
        key_states = self.rotary.apply_rotary_pos(key_states, cos, sin)
        past_kv = (key_states, value_states)

        # kv cache
        if past_key_value is not None:
            past_key, past_value = past_key_value[0], past_key_value[1]
            key_states = torch.cat((past_key, key_states), dim=1)
            value_states = torch.cat((past_value, value_states), dim=1)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        #------- attention ----------
        # query_states @ key_states
        attn_weights = torch.matmul(query_states.transpose(1, 2), key_states.transpose(1, 2).transpose(2, 3)) / math.sqrt(self.head_dim)
        # attention_mask
        attn_weights = attn_weights + attention_mask
        # upcast softmax to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights @ value_states
        attn_output = torch.matmul(attn_weights, value_states.transpose(1, 2))

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, past_kv

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class Rotary(torch.nn.Module):
    def __init__(self, config, seq_length):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.rotary_dim = config.head_dim
        self.model_type = config.model_type
        if hasattr(config, 'rotary_dim'):
            self.rotary_dim = config.rotary_dim
        if self.model_type == 'chatglm':
            self.rotary_dim = config.head_dim // 2

        self.cos, self.sin = self.init_rotary_pos_emb(seq_length)
        self.cos = self.cos.squeeze(0)
        self.sin = self.sin.squeeze(0)

    def forward(self, position_ids):
        theta = 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        if self.model_type != 'chatglm2':
            rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        return rotary_pos_emb
    
    def init_rotary_pos_emb(self, seq_length):
        position_ids = torch.tensor([range(seq_length)], dtype=torch.long)
        return self.forward(position_ids)

    def apply_rotary_pos(self, x, cos, sin):
        if self.model_type == 'chatglm':
            return self.chatglm_rotary_pos(x, cos, sin)
        if self.model_type == 'chatglm2':
            return self.chatglm2_rotary_pos(x, cos, sin)
        if self.model_type == 'phi-msft':
            return self.phi_rotary_pos(x, cos, sin)
        return self.llama_rotary_pos(x, cos, sin)

    def llama_rotary_pos(self, x, cos, sin):
        x = (x * cos) + (rotate_half(x) * sin)
        return x

    def phi_rotary_pos(self, x, cos, sin):
        x, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        x = (x * cos) + (rotate_half(x) * sin)
        return torch.cat((x, x_pass), dim=-1)

    def chatglm2_rotary_pos(self, x, cos, sin):
        x, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        b, s, n, h = x.shape
        xshaped = x.view(b, s, n, h//2, 2)
        x = torch.concat(
            [
                xshaped[..., 0] * cos - xshaped[..., 1] * sin,
                xshaped[..., 1] * cos + xshaped[..., 0] * sin,
            ],
            -1,
        )
        return torch.cat((x, x_pass), dim=-1)

    def chatglm_rotary_pos(self, x, cos, sin):
        seq = x.shape[1]
        x1, x2 = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        cos1, sin1 = cos[:, :seq, ...], sin[:, :seq, ...]
        cos2, sin2 = cos[:, seq:, ...], sin[:, seq:, ...]
        x1 = (x1 * cos1) + (rotate_half(x1) * sin1)
        x2 = (x2 * cos2) + (rotate_half(x2) * sin2)
        return torch.cat((x1, x2), dim=-1)

class Decoder(torch.nn.Module):
    def __init__(self, decoder, config):
        super().__init__()
        ModelMapper.do_map(self, decoder, config.model_map['decoder'])
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(self.self_attn, config)
        # chatglm
        self.alpha = (2 * config.num_hidden_layers) ** 0.5 if config.model_type == 'chatglm' else 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        norm_hidden_states = hidden_states
        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        # Fully Connected
        if self.alpha != 1.0:
            # chatglm-6b
            hidden_states = norm_hidden_states * self.alpha + hidden_states
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input)
            hidden_states = mlp_input * self.alpha + mlp_output
        elif hasattr(self, 'post_attention_layernorm'):
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            # phi
            feed_forward_hidden_states = self.mlp(norm_hidden_states)
            hidden_states = hidden_states + feed_forward_hidden_states + residual

        return hidden_states, present_key_value

class Lm(torch.nn.Module):
    def __init__(self, lm_, final_layernorm_, config):
        super().__init__()
        self.final_layernorm = final_layernorm_
        self.lm = lm_
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states):
        hidden_states = self.final_layernorm(hidden_states)
        m_logits = self.lm(hidden_states)
        return m_logits

class GreedyHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, m_logits):
        _, token = torch.topk(m_logits.float(), 1)
        return token

# refs:https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
class PenaltySampleHead(torch.nn.Module):

    def __init__(self, top_k = 50, min_tokens_to_keep = 5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, :self.min_tokens_to_keep] = True

    def forward(self, m_logits, input_ids, top_p, temperature, penalty):
        # repeat penalty
        logits = torch.gather(m_logits, 1, input_ids)
        logits = torch.where(logits < 0, logits * penalty, logits / penalty)
        m_logits.scatter_(1, input_ids, logits)

        # top_k
        logits, token = torch.topk(m_logits.float(), self.top_k)

        # temperature
        logits = logits / temperature

        # top_p
        cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
        mask = cumulative_probs < top_p
        mask = mask + self.keep_matrix
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token

class ModelExporter(torch.nn.Module):
    '''
    Base class for all llm model export. Inherits from [`torch.nn.Module`].
    '''

    def __init__(self, args):
        super().__init__()
        self.init_from_args(args)
        self.load_model(args.path)

        self.bmodel_converter = BmodelConverter(self.dst_path, self, args)

    def init_from_args(self, args):
        self.model_name = args.path
        self.seq_length = args.seq_length
        self.dst_path = args.dst_path

        os.makedirs(self.dst_path, exist_ok=True)
        

    def load_pretrained(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).float().eval()
        except:
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float().eval()
        self.config = self.model.config

    def rebuild_config(self):
        if not hasattr(self, 'num_key_value_heads') or self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if not hasattr(self, 'rope_theta') or self.rope_theta is None:
            self.rope_theta = 10000.0
        self.head_dim = self.hidden_size // self.num_attention_heads

    def rebuild_modules(self):
        # Embedding
        if self.embed_.weight is self.lm_.weight:
            import copy
            embed_copy = copy.deepcopy(self.embed_)
            self.embed = Embedding(embed_copy, self)
        else:
            self.embed = Embedding(self.embed_, self)
        # Rotary
        self.rotary = Rotary(self, self.seq_length)
        # Blocks
        self.blocks = []
        for block in self.blocks_.children():
            self.blocks.append(Decoder(block, self))
        # Lmhead
        self.lm = Lm(self.lm_, self.final_layernorm_, self)

    def load_model(self, model_path):
        self.load_pretrained(model_path)

        model_mapper = ModelMapper()
        self.model_type, self.model_map = model_mapper.get_map(self.model.config)

        # load config
        ModelMapper.do_map(self, self.model.config, self.model_map['config'])
        # rebuild config
        self.rebuild_config()

        # load modules
        ModelMapper.do_map(self, self.model, self.model_map['model'])
        # rebuild modules
        self.rebuild_modules()
        return model_path
    
    @logging("export_config ...")
    def export_config(self):
        config_dict = self.config.to_dict()
        with open(f'{self.dst_path}/config.json', "w") as f:
            json.dump(config_dict, f, indent=4)
        return
    
    @logging("export_tokenizer ...")
    def export_tokenizer(self):
        self.tokenizer.save_pretrained(f'{self.dst_path}/tokenizer')
        return

    @logging("export_embed ...")
    def export_embed(self):
        if not hasattr(self, 'embed') or not isinstance(self.embed.embed, torch.nn.Embedding):
            return

        embedding_file = f'{self.dst_path}/embedding.bin'
        if os.path.exists(embedding_file):
            print(f"{embedding_file} already exists. Skipping export.")
            return

        import ctypes
        if self.config.torch_dtype == torch.bfloat16:
            tensor_data = self.embed.embed.weight.data.bfloat16()
        else:
            raise ValueError("not support now")
        data_ptr = tensor_data.untyped_storage().data_ptr()
        buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
        
        with open(embedding_file, 'wb') as f:
            f.write(buffer)
        return

    @logging("export_block ...")
    def export_block(self):
        hidden_states = torch.randn((1, self.seq_length, self.hidden_size), dtype=torch.float)
        position_ids = torch.tensor([range(self.seq_length)], dtype=torch.long)
        attention_mask = torch.randn((1, 1, self.seq_length, self.seq_length), dtype=torch.float)

        rebuilder = OnnxRebuild()
        for i in tqdm(range(len(self.blocks))):
            model = self.blocks[i]
            onnx_path = f'{self.dst_path}/onnx/block_{i}.onnx'
            if os.path.exists(onnx_path):
                print(f"{onnx_path} already exists. Skipping export.")
                continue

            if i == 0:
                torch.onnx.export(
                    model,
                    (hidden_states, position_ids, attention_mask),
                    onnx_path,
                    verbose=False,
                    input_names=["input_states", "position_ids", "attention_mask"],
                    output_names=["hidden_states", "past_k", "past_v"],
                    do_constant_folding=False, # set False to keep original name
                    opset_version=15,
                )
            else:
                rebuilder.rebuild_weights(
                    ref_path=f'{self.dst_path}/onnx/block_0.onnx', 
                    torch_model=model,
                    save_path=onnx_path
                )
        return

    @logging("export_block_cache ...")
    def export_block_cache(self):
        hidden_states = torch.randn((1, self.seq_length, self.hidden_size), dtype=torch.float)
        position_ids = torch.tensor([range(self.seq_length)], dtype=torch.long)
        attention_mask = torch.randn((1, 1, self.seq_length, self.seq_length), dtype=torch.float)

        hidden_states = torch.randn((1, 1, self.hidden_size), dtype=torch.float)
        position_ids = torch.tensor([range(1)], dtype=torch.long)
        attention_mask = torch.ones(
            (1, 1, 1, self.seq_length + 1))
        past_k = torch.randn((1, self.seq_length, self.num_key_value_heads, self.head_dim), dtype=torch.float)
        past_v = torch.randn((1, self.seq_length, self.num_key_value_heads, self.head_dim), dtype=torch.float)

        rebuilder = OnnxRebuild()
        for i in tqdm(range(len(self.blocks))):
            model = self.blocks[i]
            onnx_path = f'{self.dst_path}/onnx/block_cache_{i}.onnx'
            if os.path.exists(onnx_path):
                print(f"{onnx_path} already exists. Skipping export.")
                continue

            if i == 0:
                torch.onnx.export(
                    model,
                    (hidden_states, position_ids, attention_mask, (past_k, past_v)),
                    onnx_path,
                    verbose=False,
                    input_names=["input_states", "position_ids", "attention_mask", "history_k", "history_v"],
                    output_names=["hidden_states", "past_k", "past_v"],
                    do_constant_folding=False,
                    opset_version=15,
                )
            else:
                rebuilder.rebuild_weights(
                    ref_path=f'{self.dst_path}/onnx/block_cache_0.onnx', 
                    torch_model=model,
                    save_path=onnx_path
                )
        return
    
    @logging("export_lm_head ...")
    def export_lm_head(self):
        pt_path = f'{self.dst_path}/onnx/lm_head.pt'
        if os.path.exists(pt_path):
            print(f"{pt_path} already exists. Skipping export.")
            return

        model = self.lm
        hidden_states = torch.randn((1, 1, self.hidden_size), dtype=torch.float)
        module = torch.jit.trace(model.forward, hidden_states)
        torch.jit.save(module, pt_path)
        return

    def export_greedy_head(self):
        onnx_path = f'{self.dst_path}/onnx/greedy_head.onnx'
        if os.path.exists(onnx_path):
            print(f"{onnx_path} already exists. Skipping export.")
            return

        model = GreedyHead()
        m_logits = torch.randn(1, self.config.vocab_size)
        torch.onnx.export(
            model, (m_logits),
            onnx_path,
            verbose=False,
            input_names=['m_logits'],
            output_names=['token'],
            do_constant_folding=True,
            opset_version=15)
        return

    def export_penalty_sample_head(self):
        onnx_path = f'{self.dst_path}/onnx/penalty_sample_head.onnx'
        if os.path.exists(onnx_path):
            print(f"{onnx_path} already exists. Skipping export.")
            return

        model = PenaltySampleHead()
        m_logits = torch.randn(1, self.config.vocab_size)
        input_ids = torch.tensor([range(self.seq_length)])
        top_p = torch.tensor([0.8])
        temperature = torch.tensor([0.98])
        penalty = torch.tensor([0.98])

        torch.onnx.export(
            model, (m_logits, input_ids, top_p, temperature, penalty),
            onnx_path,
            verbose=False,
            input_names=[
                'm_logits', 'input_ids', 'top_p', 'temperature',
                'penalty'
            ],
            output_names=['probs', 'token'],
            do_constant_folding=True,
            opset_version=15)
        return

    def export_onnx(self):
        # mkir
        onnx_path = os.path.join(self.dst_path, "onnx")
        os.makedirs(onnx_path, exist_ok=True)

        self.export_config()
        self.export_tokenizer()
        self.export_embed()
        self.export_block()
        self.export_block_cache()
        self.export_lm_head()
        self.export_greedy_head()
        self.export_penalty_sample_head()

    def check(self, args):
        if args.seq_length is None:
            raise ValueError("Please provide a value for --seq_length, when using the --export option.")
        if args.export == "bmodel":
            if args.tpu_mlir_path is None:
                raise ValueError("Please provide a path for --tpu_mlir_path, when using the --export bmodel.")

    def export(self, export_type):

        self.export_onnx()
        if 'bmodel' in export_type:
            self.bmodel_converter.compile()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--path', type=str, required=True,
                        help='path(`str` or `os.PathLike`):\nCan be either:'
                        '\n\t- A string, the *model id* of a pretrained model like `THUDM/chatglm-6b`. [TODO]'
                        '\n\t- A path to a *directory* clone from repo like `../chatglm-6b`.')
    parser.add_argument('--export', type=str, choices=["onnx", "bmodel"], default=None, help='export torch/onnx to an onnx/bmodel model.')
    parser.add_argument('--dst_path', type=str, default='./model', help='export onnx/bmodel model to path, defaut is `./model`.')
    parser.add_argument('--bmodel', type=str, default='', help='bmodel name after model_tool --combine')
    parser.add_argument('--seq_length', type=int, required=True, help="sequence length")
    parser.add_argument('--chip', type=str, default="bm1684x", choices=["bm1684x", "bm1688"], help="chip")
    parser.add_argument('--quantize', type=str, default="w4bf16", choices=["bf16", "w8bf16", "w4bf16", "f16", "w8f16", "w4f16"], help="quantize")
    parser.add_argument('--num_device', type=int, default=1, help="num device in compiling bmodel")
    parser.add_argument('--max_workers', type=int, default=3, help="max workers for compiling bmodel in multi-processing")
    parser.add_argument('--tpu_mlir_path', type=str, help="tpu_mlir for compiling bmodel")
    args = parser.parse_args()

    model_exporter = ModelExporter(args)

    if args.export is not None:
        model_exporter.check(args)
        model_exporter.export(args.export)