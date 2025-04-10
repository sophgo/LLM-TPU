#!/usr/bin/env python3
import mlir
from mlir.ir import *
import mlir.dialects.top as top
import numpy as np
import re
from llm_info import *
import os
import torch
import concurrent.futures
import subprocess
from datetime import datetime
from safetensors import safe_open
import sys
from tqdm import tqdm
from transformers import AutoConfig


class MlirRebuilder:

    def __init__(self, name: str, context: str, out_dir: str, reuse_weight_name: str = None):
        self.name = name
        self.output_file = os.path.join(out_dir, f"{name}.mlir")
        self.weight_reuse = False
        if reuse_weight_name:
            self.weight_reuse = True
            self.weight_file_name = f"{reuse_weight_name}_top_weight.npz"
        else:
            self.weight_file_name = f"{name}_top_weight.npz"
        self.weight_file = os.path.join(out_dir, self.weight_file_name)
        self.ctx = mlir.ir.Context()
        self.ctx.allow_unregistered_dialects = True
        self.module = mlir.ir.Module.parse(context, self.ctx)
        self.body = self.module.body.operations[0].regions[0].blocks[0]
        self.set_attr(self.module.operation, 'sym_name', StringAttr.get(name, self.ctx))
        self.set_attr(self.module.operation, 'module.weight_file',
                      StringAttr.get(self.weight_file_name, self.ctx))
        self.weight_ops = []
        for op in self.body.operations:
            if isinstance(op, top.WeightOp):
                self.weight_ops.append(op)
        self.weight_datas = {}

    def set_attr(self, op: Operation, name: str, value):
        op.attributes.__setitem__(name, value)

    def get_op_name(self, op: Operation):
        loc = str(op.location)
        name = re.search(r'loc\(\"(.+?)\"\)', loc).group(1)
        return name

    def get_num_weight(self):
        return len(self.weight_ops)

    def get_weight_name(self, idx: int):
        assert (len(self.weight_ops) > idx)
        wop = self.weight_ops[idx]
        wname = self.get_op_name(wop)
        return wname

    def set_weight(self, idx: int, weight: np.ndarray):
        assert (self.weight_reuse == False)
        assert (len(self.weight_ops) > idx)
        wop = self.weight_ops[idx]
        wname = self.get_op_name(wop)
        shape = wop.output.type.shape
        num_elems = np.prod(shape)
        if num_elems != weight.size:
            print(f"Error: Weight[{idx}] size mismatch: {num_elems} vs {weight.size}")
            breakpoint()
        weight.reshape(shape)
        self.weight_datas[wname] = weight

    def save(self):
        module_txt = self.module.operation.get_asm(enable_debug_info=True)
        with open(self.output_file, "w") as f:
            f.write(module_txt)
        if not self.weight_reuse:
            np.savez(self.weight_file, **self.weight_datas)
        tqdm.write(f"Success: Saved to {self.output_file}")


class LlmLoad:

    def __init__(self, model_path: str):
        self.st_files = []
        # get all safetensors
        for entry in os.listdir(model_path):
            file_path = os.path.join(model_path, entry)
            if os.path.isfile(file_path) and entry.lower().endswith('.safetensors'):
                f = safe_open(file_path, "pt")
                self.st_files.append(f)

    def read(self, key: str):
        for f in self.st_files:
            if key in f.keys():
                data = f.get_tensor(key)
                if data.dtype in [torch.float16, torch.bfloat16]:
                    return data.float().numpy()
                return data.numpy()
        raise RuntimeError(f"Can't find key: {key}")


class LlmConvert:

    def __init__(self, args):
        self.model_path = args.model_path
        self.seq_length = args.seq_length
        self.quantize = args.quantize
        self.num_device = args.num_device
        self.q_group_size = args.q_group_size
        self.high_precision = True
        self.symmetric = args.symmetric
        self.lmhead_with_topk = args.num_device > 1
        self.tpu_mlir_path = args.tpu_mlir_path
        self.chip = args.chip
        self.num_device = args.num_device
        self.embedding_disk = args.embedding_disk
        self.num_core = args.num_core if args.chip == "bm1688" else 1
        self.half_precision_quantize = "bf16" if "bf16" in self.quantize else "f16"
        self.load_pretrained()
        # get attributes
        self.init_config()
        cos, sin = self.get_rotary_pos_emb(self.seq_length)
        self.cos = cos.numpy()
        self.sin = sin.numpy()
        cpu_count = os.cpu_count()
        self.max_workers = max(cpu_count, 4)
        # read mlir context
        self.context_readall()
        # get file path
        self.out_dir = args.out_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.chip == "bm1684x":
            folder_name = f"bmodel_seq{self.seq_length}_{self.quantize}_{self.chip}_{self.num_device}dev"
            self.out_bmodel = f"../{self.model_type}_{self.quantize}_seq{self.seq_length}_{self.chip}_{self.num_device}dev_{timestamp}.bmodel"
        else:
            folder_name = f"bmodel_seq{self.seq_length}_{self.quantize}_{self.chip}_{self.num_core}core"
            self.out_bmodel = f"../{self.model_type}_{self.quantize}_seq{self.seq_length}_{self.chip}_{self.num_core}core_{timestamp}.bmodel"

        self.bmodel_dir = os.path.join(self.out_dir, folder_name)
        self.commands = []

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.bmodel_dir, exist_ok=True)
        # export mlir files
        self.gen_all_mlir()
        del self.model
        self.compile_all()

    def init_config(self):
        c = self.model_info.config
        self.num_layers = getattr(self.config, c.num_hidden_layers)
        self.rope_theta = getattr(self.config, c.rope_theta, 10000.0)
        self.num_attention_heads = getattr(self.config, c.num_attention_heads)
        self.num_key_value_heads = getattr(self.config, c.num_key_value_heads,
                                           self.num_attention_heads)
        self.hidden_size = getattr(self.config, c.hidden_size)
        self.vocab_size = getattr(self.config, c.vocab_size)
        self.intermediate_size = getattr(self.config, c.intermediate_size)
        self.rms_norm_eps = getattr(self.config, c.rms_norm_eps)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rotary_dim = self.head_dim
        if hasattr(self.config, 'rotary_dim'):
            self.rotary_dim = self.config.rotary_dim
        if self.model_type == 'chatglm':
            self.rotary_dim = self.config.head_dim // 2

    def replace_template(self, template: str, replace_dict: dict) -> str:
        keys = map(re.escape, replace_dict.keys())
        pattern = r"\{(" + "|".join(keys) + r")\}"

        def replacer(match):
            key = match.group(1)
            return str(replace_dict[key])

        return re.sub(pattern, replacer, template)

    def context_format(self, context: str):
        replace_dict = {
            "hidden_size": self.hidden_size,
            "seq_length": self.seq_length,
            "seq_next": self.seq_length + 1,
            "vocab_size": self.vocab_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "kv_dim": self.num_key_value_heads * self.head_dim,
            "half_head_dim": self.head_dim // 2,
            "scaling": self.head_dim**-0.5,
            "kv_tile": self.num_attention_heads // self.num_key_value_heads,
            "rms_norm_eps": f"{self.rms_norm_eps:.9f}",
        }
        return self.replace_template(context, replace_dict)

    def context_read(self, mlir_file: str):
        with open(mlir_file, "r") as f:
            context = f.read()
            return self.context_format(context)

    def context_readall(self):
        self.embed_mlir = "models/common/embedding.mlir"
        self.embed2_mlir = "models/common/embedding_cache.mlir"
        if self.lmhead_with_topk:
            self.lmhead_mlir = "models/common/lm_head_with_topk.mlir"
        else:
            self.lmhead_mlir = "models/common/lm_head.mlir"
            self.ghead_mlir = "models/common/greedy_head.mlir"
            self.phead_mlir = "models/common/penalty_sample_head.mlir"

        self.embed_context = self.context_read(self.embed_mlir)
        self.embed2_context = self.context_read(self.embed2_mlir)
        self.block_context = self.context_read(self.block_mlir)
        self.block_cache_context = self.context_read(self.block_cache_mlir)
        self.lmhead_context = self.context_read(self.lmhead_mlir)
        if not self.lmhead_with_topk:
            self.ghead_context = self.context_read(self.ghead_mlir)
            self.phead_context = self.context_read(self.phead_mlir)

    def load_pretrained(self):
        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = LlmLoad(self.model_path)
        self.model_type = self.config.model_type
        if 'qwen2' == self.model_type:
            self.model_info = QWEN2_INFO
            self.block_mlir = "models/qwen2/block_0.mlir"
            self.block_cache_mlir = "models/qwen2/block_cache_0.mlir"
        else:
            raise RuntimeError("Not Implemented")

    def get_rotary_pos_emb(self, seq_length):
        position_ids = torch.tensor([range(seq_length)], dtype=torch.long)
        theta = 1.0 / (self.rope_theta**(torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) /
                                         self.rotary_dim))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        if self.model_type != 'chatglm2':
            rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        return rotary_pos_emb

    def read_weight(self, weight: WeightInfo, obj: str = ""):
        if obj:
            name = f"{obj}.{weight.name}"
        else:
            name = weight.name
        if weight.type == WeightType.MM_BIAS:
            return self.model.read(name + ".bias")
        if weight.type == WeightType.MM_WEIGHT:
            data = self.model.read(name + ".weight")
            return np.ascontiguousarray(np.transpose(data, (1, 0)))
        return self.model.read(name)

    def export_block(self, layer_idx: int):
        tqdm.write(f"export block {layer_idx}")
        block_builder = MlirRebuilder(f"block_{layer_idx}", self.block_context, self.bmodel_dir)
        block_cache_builder = MlirRebuilder(f"block_cache_{layer_idx}", self.block_cache_context,
                                            self.bmodel_dir, f"block_{layer_idx}")
        num_weight = len(self.model_info.weights.blocks)
        assert (block_builder.get_num_weight() == num_weight)
        assert (block_cache_builder.get_num_weight() == num_weight)
        for widx in range(num_weight):
            if block_builder.get_weight_name(widx) != block_cache_builder.get_weight_name(widx):
                print("Error: weight name mismatch")
                breakpoint()
            weight = self.model_info.weights.blocks[widx]
            if weight.type == WeightType.ROTARY_COS:
                data = self.cos
            elif weight.type == WeightType.ROTARY_SIN:
                data = self.sin
            else:
                obj = f"{self.model_info.weights.layers}.{layer_idx}"
                data = self.read_weight(weight, obj)
            block_builder.set_weight(widx, data)
        block_builder.save()
        block_cache_builder.save()

    def export_lmhead(self):
        tqdm.write("export lm_head")
        norm_weight = self.read_weight(self.model_info.weights.norm)
        lmhead_builder = MlirRebuilder("lm_head", self.lmhead_context, self.bmodel_dir)
        lmhead_builder.set_weight(0, norm_weight)
        lmhead_weight = self.read_weight(self.model_info.weights.lm_head)
        lmhead_builder.set_weight(1, lmhead_weight)
        lmhead_builder.save()
        if not self.lmhead_with_topk:
            ghead_builder = MlirRebuilder("greedy_head", self.ghead_context, self.bmodel_dir)
            ghead_builder.save()
            phead_builder = MlirRebuilder("penalty_sample_head", self.phead_context,
                                          self.bmodel_dir)
            cumsum = np.array([1.0], dtype=np.float32)
            phead_builder.set_weight(0, cumsum)
            keep_matrix = np.zeros((1, 50), dtype=np.float32)
            keep_matrix[0, :5] = 1.0
            phead_builder.set_weight(1, keep_matrix)
            phead_builder.save()

    def export_embedding(self):
        tqdm.write("export embedding")
        if not self.embedding_disk:
            embed_builder = MlirRebuilder("embedding", self.embed_context, self.bmodel_dir)
            embed2_builder = MlirRebuilder("embedding_cache", self.embed2_context, self.bmodel_dir,
                                           "embedding")
            weight = self.read_weight(self.model_info.weights.embed)
            embed_builder.set_weight(0, weight)
            embed_builder.save()
            embed2_builder.save()
        else:
            embedding_file = f'{self.out_dir}/embedding.bin'
            if os.path.exists(embedding_file):
                print(f"{embedding_file} already exists. Skipping export.")
                return
            import ctypes
            weight = self.read_weight(self.model_info.weights.embed)
            weight = torch.from_numpy(weight)
            if 'bf16' in self.quantize:
                tensor_data = weight.to(torch.bfloat16)
            elif 'f16' in self.quantize:
                tensor_data = weight.to(torch.float16)
            else:
                raise NotImplementedError("Not support now")
            data_ptr = tensor_data.untyped_storage().data_ptr()
            buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
            with open(embedding_file, 'wb') as f:
                f.write(buffer)

    def gen_all_mlir(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []

            for i in range(self.num_layers):
                futures.append(executor.submit(self.export_block, i))

            futures.append(executor.submit(self.export_lmhead))
            futures.append(executor.submit(self.export_embedding))

            # Wait for all threads to complete
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               desc="Exporting Blocks"):
                # This will raise exceptions if any occurred during thread execution
                future.result()

    def send_command(self, command: list[str], log_file: str):
        command.append(f"> {log_file}\n")
        cmd = ' '.join(command)
        self.commands.append(cmd)

    def run_command(self, command):
        GREEN_COLOR = "\033[92m"  # ANSI escape code for green text
        RED_COLOR = "\033[91m"
        RESET_COLOR = "\033[0m"
        try:
            print(f"{GREEN_COLOR}Executing command: \n{' '.join(command)}{RESET_COLOR}"
                  )  # Print the command in green
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            # Print the error message in red
            print(f"{RED_COLOR}Error: Command failed with return code {e.returncode}{RESET_COLOR}")
            print(f"{RED_COLOR}Failed command: {' '.join(command)}{RESET_COLOR}")
            # Exit the program with the same return code as the failed command
            sys.exit(e.returncode)

    def execute_commands(self):
        task_file = "task.txt"
        with open(task_file, "w") as f:
            f.writelines(self.commands)
        self.commands.clear()
        parallel_cmd = [
            "parallel", f"-j {self.max_workers}", "--progress", f"--joblog {task_file}.log",
            f"< {task_file}"
        ]
        self.run_command(['bash', '-c', ' '.join(parallel_cmd)])

    def compile_embedding(self):
        name = "embedding"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.half_precision_quantize}',
            '--quant_input', '--quant_output', f'--chip {self.chip}', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        self.send_command(deploy_args, f"{name}.log")

    def compile_embedding_cache(self):
        name = "embedding_cache"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.half_precision_quantize}',
            '--quant_input', '--quant_output', f'--chip {self.chip}', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        self.send_command(deploy_args, f"{name}.log")

    def compile_lm_head(self):
        name = "lm_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.half_precision_quantize}',
            '--quant_input', f'--chip {self.chip}', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        self.send_command(deploy_args, f"{name}.log")

    def compile_greedy_head(self):
        name = "greedy_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--model {name}.bmodel'
        ]
        self.send_command(deploy_args, f"{name}.log")

    def compile_penalty_head(self):
        name = "penalty_sample_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--model {name}.bmodel'
        ]
        self.send_command(deploy_args, f"{name}.log")

    def compile_block(self, layer_id):
        name = f"block_{layer_id}"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return

        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.quantize}',
            f'--q_group_size {self.q_group_size}', '--quant_input', '--quant_output',
            f'--chip {self.chip}', f'--num_core {self.num_core}', f'--num_device {self.num_device}',
            f'--model {name}.bmodel'
        ]
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.symmetric:
            deploy_args.append('--q_symmetric')
        self.send_command(deploy_args, f"{name}.log")

    def compile_block_cache(self, layer_id):
        name = f"block_cache_{layer_id}"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return

        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.quantize}',
            f'--q_group_size {self.q_group_size}', '--quant_input', '--quant_output',
            f'--chip {self.chip}', '--addr_mode io_alone', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.symmetric:
            deploy_args.append('--q_symmetric')
        self.send_command(deploy_args, f"{name}.log")

    def combine(self):
        bmodel_list = []
        for i in range(self.num_layers):
            bmodel_list = bmodel_list + [f"block_{i}.bmodel", f"block_cache_{i}.bmodel"]
        if not self.embedding_disk:
            bmodel_list += ['embedding.bmodel', 'embedding_cache.bmodel']
        if not self.lmhead_with_topk:
            bmodel_list += ["greedy_head.bmodel", "penalty_sample_head.bmodel"]
        bmodel_list += ["lm_head.bmodel"]

        combine_args = ['model_tool', '--combine', ' '.join(bmodel_list), '-o', self.out_bmodel]
        self.run_command(['bash', '-c', ' '.join(combine_args)])

        get_info_args = ['model_tool', '--info', self.out_bmodel, '> ../model.log']
        self.run_command(['bash', '-c', ' '.join(get_info_args)])

    def compile_all(self):
        ori_path = os.getcwd()
        os.chdir(self.bmodel_dir)

        if not self.embedding_disk:
            self.compile_embedding()
            self.compile_embedding_cache()

        self.compile_lm_head()

        if not self.lmhead_with_topk:
            self.compile_greedy_head()
            self.compile_penalty_head()

        for i in range(self.num_layers):
            self.compile_block(i)
            self.compile_block_cache(i)

        self.execute_commands()

        # Combine all bmodel files
        self.combine()

        # Remove any .npz files
        for npz_file in os.listdir():
            if os.path.splitext(npz_file)[-1] == '.npz':
                os.remove(npz_file)

        # Change back to the original directory
        os.chdir(ori_path)
