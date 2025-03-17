#!/usr/bin/env python3
import os
import sys
import time
import warnings
import logging
import argparse
import subprocess
import concurrent.futures
from datetime import datetime

warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("onnx").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
    AutoConfig,
)
from onnx_rebuilder import *

GREEN_COLOR = "\033[92m"  # ANSI escape code for green text
RED_COLOR = "\033[91m"
RESET_COLOR = "\033[0m"

class BmodelConverter:
    def __init__(self,
                 args,
                 hidden_size: int,
                 num_layers: int,
                 model_type: str,
                 bmodel_dir: str,
                 visual: bool):
        self.chip = args.chip
        self.quantize = args.quantize
        self.seq_length = args.seq_length
        self.num_device = args.num_device
        self.max_workers = args.max_workers
        self.compile_mode = args.compile_mode
        self.tpu_mlir_path = args.tpu_mlir_path
        self.embedding_disk = args.embedding_disk
        self.half_precision_quantize = "bf16" if "bf16" in self.quantize else "f16"

        self.bmodel_dir = bmodel_dir
        self.relative_onnx_path = "../onnx"
        if self.chip == "bm1688":
            self.num_core = 2
        else:
            self.num_core = 1

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.visual = visual

        if args.out_bmodel:
            self.out_bmodel = '../' + args.out_bmodel
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.out_bmodel = f"../{model_type}_{self.quantize}_seq{self.seq_length}_{self.chip}_{timestamp}.bmodel"

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
        try:
            print(f"{GREEN_COLOR}Executing command: \n{' '.join(command)}{RESET_COLOR}")  # Print the command in green
            subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError as e:
            # Print the error message in red
            print(f"{RED_COLOR}Error: Command failed with return code {e.returncode}{RESET_COLOR}")
            print(f"{RED_COLOR}Failed command: {' '.join(command)}{RESET_COLOR}")
            # Exit the program with the same return code as the failed command
            sys.exit(e.returncode)

    def compile_embedding(self, env):
        name = "embedding"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
        else:
            path = os.path.join(self.relative_onnx_path, f"{name}.pt")
            if self.compile_mode == 'fast':
                convert_args = [
                    'model_convert.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--input_shapes [[1,{self.seq_length}]]',
                    f'--input_types "int32"',
                    f'--quantize {self.quantize}',
                    '--quant_input',
                    '--quant_output',
                    f'--chip {self.chip}',
                    f'--num_core {self.num_core}',
                    f'--num_device {self.num_device}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(['bash', '-c', ' '.join(convert_args)], env)
            else:
                transform_args = [
                    'model_transform.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--input_shapes [[1,{self.seq_length}]]',
                    f'--input_types "int32"',
                    f'--mlir {name}.mlir'
                ]
                deploy_args = [
                    'model_deploy.py',
                    f'--mlir {name}.mlir',
                    f'--quantize {self.quantize}',
                    '--quant_input',
                    '--quant_output',
                    f'--chip {self.chip}',
                    f'--num_core {self.num_core}',
                    f'--num_device {self.num_device}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(['bash', '-c', ' '.join(transform_args)], env)
                self.run_command(['bash', '-c', ' '.join(deploy_args)], env)

    def compile_embedding_cache(self, env):
        name = "embedding_cache"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
        else:
            path = os.path.join(self.relative_onnx_path, f"embedding.pt")
            if self.compile_mode == 'fast':
                convert_args = [
                    'model_convert.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--input_shapes [[1,1]]',
                    f'--input_types "int32"',
                    f'--quantize {self.quantize}',
                    '--quant_input',
                    '--quant_output',
                    f'--chip {self.chip}',
                    f'--num_core {self.num_core}',
                    f'--num_device {self.num_device}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(['bash', '-c', ' '.join(convert_args)], env)
            else:
                transform_args = [
                    'model_transform.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--input_shapes [[1,1]]',
                    f'--input_types "int32"',
                    f'--mlir {name}.mlir'
                ]
                deploy_args = [
                    'model_deploy.py',
                    f'--mlir {name}.mlir',
                    f'--quantize {self.quantize}',
                    '--quant_input',
                    '--quant_output',
                    f'--chip {self.chip}',
                    f'--num_core {self.num_core}',
                    f'--num_device {self.num_device}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(['bash', '-c', ' '.join(transform_args)], env)
                self.run_command(['bash', '-c', ' '.join(deploy_args)], env)

    def compile_lm_head(self, env):
        name = "lm_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
        else:
            path = os.path.join(self.relative_onnx_path, f"{name}.pt")
            if self.compile_mode == 'fast':
                convert_args = [
                    'model_convert.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--input_shapes [[1,{self.hidden_size}]]',
                    f'--quantize {self.half_precision_quantize}',
                    '--quant_input',
                    f'--chip {self.chip}',
                    f'--num_core {self.num_core}',
                    f'--num_device {self.num_device}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(['bash', '-c', ' '.join(convert_args)], env)
            else:
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
                    f'--quantize {self.half_precision_quantize}',
                    '--quant_input',
                    f'--chip {self.chip}',
                    f'--num_core {self.num_core}',
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
            if self.compile_mode == 'fast':
                convert_args = [
                    'model_convert.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--chip {self.chip}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(['bash', '-c', ' '.join(convert_args)], env)
            else:
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
            if self.compile_mode == 'fast':
                convert_args = [
                    'model_convert.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--chip {self.chip}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(['bash', '-c', ' '.join(convert_args)], env)
            else:
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

    def compile_block(self, layer_id, env):
        name = f"block_{layer_id}"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
        else:
            path = os.path.join(self.relative_onnx_path, f"{name}.onnx")
            if self.compile_mode == 'fast':
                convert_args = [
                    'model_convert.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--quantize {self.quantize}',
                    '--quant_input',
                    '--quant_output',
                    '--do_onnx_sim True',
                    f'--chip {self.chip}',
                    f'--num_core {self.num_core}',
                    f'--num_device {self.num_device}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(['bash', '-c', ' '.join(convert_args)], env)
            else:
                transform_args = [
                    'model_transform.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--mlir {name}.mlir'
                ]
                deploy_args = [
                    'model_deploy.py',
                    f'--mlir {name}.mlir',
                    f'--quantize {self.quantize}',
                    '--quant_input',
                    '--quant_output',
                    f'--chip {self.chip}',
                    f'--num_core {self.num_core}',
                    f'--num_device {self.num_device}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(['bash', '-c', ' '.join(transform_args)], env)
                self.run_command(['bash', '-c', ' '.join(deploy_args)], env)

    def compile_block_cache(self, layer_id, env):
        name = f"block_cache_{layer_id}"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
        else:
            path = os.path.join(self.relative_onnx_path, f"{name}.onnx")
            if self.compile_mode == 'fast':
                convert_args = [
                    'model_convert.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--quantize {self.quantize}',
                    '--quant_input',
                    '--quant_output',
                    '--do_onnx_sim True',
                    f'--chip {self.chip}',
                    '--addr_mode io_alone',
                    f'--num_core {self.num_core}',
                    f'--num_device {self.num_device}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(
                    ['bash', '-c', ' '.join(convert_args)], env)
            else:
                transform_args = [
                    'model_transform.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--mlir {name}.mlir'
                ]
                deploy_args = [
                    'model_deploy.py',
                    f'--mlir {name}.mlir',
                    f'--quantize {self.quantize}',
                    '--quant_input',
                    '--quant_output',
                    f'--chip {self.chip}',
                    '--addr_mode io_alone',
                    f'--num_core {self.num_core}',
                    f'--num_device {self.num_device}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(
                    ['bash', '-c', ' '.join(transform_args)], env)
                self.run_command(
                    ['bash', '-c', ' '.join(deploy_args)], env)

    def compile_vit(self, env):
        name = "vit"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
        else:
            path = os.path.join(self.relative_onnx_path, "vit", f"{name}.onnx")
            if self.compile_mode == 'fast':
                convert_args = [
                    'model_convert.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--quantize {self.half_precision_quantize}',
                    '--quant_output',
                    '--do_onnx_sim True',
                    f'--chip {self.chip}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(
                    ['bash', '-c', ' '.join(convert_args)], env)
            else:
                transform_args = [
                    'model_transform.py',
                    f'--model_name {name}',
                    f'--model_def {path}',
                    f'--mlir {name}.mlir'
                ]
                deploy_args = [
                    'model_deploy.py',
                    f'--mlir {name}.mlir',
                    f'--quantize {self.half_precision_quantize}',
                    '--quant_output',
                    f'--chip {self.chip}',
                    f'--num_core {self.num_core}',
                    f'--num_device {self.num_device}',
                    f'--model {name}.bmodel'
                ]
                self.run_command(
                    ['bash', '-c', ' '.join(transform_args)], env)
                self.run_command(
                    ['bash', '-c', ' '.join(deploy_args)], env)

    def combine(self, env):
        bmodel_list = []
        for i in range(self.num_layers):
            bmodel_list = bmodel_list + [f"block_{i}.bmodel", f"block_cache_{i}.bmodel"]
        if not self.embedding_disk:
            bmodel_list += ['embedding.bmodel', 'embedding_cache.bmodel']
        if self.visual:
            bmodel_list += ["vit.bmodel"]
        if self.num_device == 1:
            bmodel_list += ["greedy_head.bmodel", "penalty_sample_head.bmodel"]
        bmodel_list += ["lm_head.bmodel"]

        combine_args = [
            'model_tool',
            '--combine',
            ' '.join(bmodel_list),
            '-o',
            self.out_bmodel
        ]
        self.run_command(['bash', '-c', ' '.join(combine_args)], env)
        
        get_info_args = [
            'model_tool',
            '--info',
            self.out_bmodel,
            '> ../model.log'
        ]
        self.run_command(['bash', '-c', ' '.join(get_info_args)], env)

    def compile(self):
        ori_path = os.getcwd()
        os.chdir(self.bmodel_dir)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            if not self.embedding_disk:
                futures.append(executor.submit(self.compile_embedding, self.env))
                futures.append(executor.submit(self.compile_embedding_cache, self.env))

            futures.append(executor.submit(self.compile_lm_head, self.env))

            if self.num_device == 1:
                futures.append(executor.submit(self.compile_greedy_head, self.env))
                futures.append(executor.submit(self.compile_penalty_head, self.env))

            if self.visual:
                futures.append(executor.submit(self.compile_vit, self.env))

            for i in range(self.num_layers):
                futures.append(executor.submit(self.compile_block, i, self.env))
                futures.append(executor.submit(self.compile_block_cache, i, self.env))

            # Wait for all threads to complete
            for future in concurrent.futures.as_completed(futures):
                # This will raise exceptions if any occurred during thread execution
                future.result()

        # Combine all bmodel files
        self.combine(self.env)

        # Remove any .npz files
        for npz_file in os.listdir():
            if os.path.splitext(npz_file)[-1] == '.npz':
                os.remove(npz_file)

        # Change back to the original directory
        os.chdir(ori_path)

class ModelExporter:

    def __init__(self, args):
        super().__init__()
        self.model_path = args.model_path
        self.seq_length = args.seq_length
        self.quantize = args.quantize
        self.not_compile = args.not_compile
        self.embedding_disk = args.embedding_disk
        self.lmhead_with_topk = args.num_device > 1
        self.out_dir = args.out_dir
        self.onnx_dir = os.path.join(self.out_dir, "onnx")
        self.bmodel_dir = os.path.join(self.out_dir, "bmodel")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.onnx_dir, exist_ok=True)
        os.makedirs(self.bmodel_dir, exist_ok=True)

        # for vision language model
        self.visual = None
        self.visual_model = None
        self.visual_length = args.visual_length

        # load original weight, save config and tokenizer
        self.load_pretrained()

        # rebuild original weight to onnx model
        self.onnx_rebuilder = OnnxRebuilder(self.onnx_dir,
                                            self.model_path,
                                            self.quantize,
                                            self.seq_length,
                                            self.model_type,
                                            self.embedding_disk,
                                            self.lmhead_with_topk,
                                            self.config,
                                            self.visual_length)
        self.rebuild_model()

        # compile bmodel
        if not self.not_compile:
            assert(args.tpu_mlir_path is not None), "Please provide a path for --tpu_mlir_path if you need compile bmodel, else use --not_compile"
            assert(args.quantize is not None), "Please provide a value for --quantize if you need compile bmodel, else use --not_compile"
            self.bmodel_converter = BmodelConverter(args,
                                                    self.config.hidden_size,
                                                    self.config.num_hidden_layers,
                                                    self.model_type,
                                                    self.bmodel_dir,
                                                    self.visual_length)

    def load_pretrained(self):
        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.model_type = self.config.model_type

        if 'qwen2_vl' == self.model_type:
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path)
        elif 'qwen2_5_vl' == self.model_type:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path)
        elif 'mllama' == self.model_type:
            from transformers import MllamaForConditionalGeneration
            self.model = MllamaForConditionalGeneration.from_pretrained(self.model_path)
        elif 'llama' == self.model_type:
            from transformers import LlamaForCausalLM
            self.model = LlamaForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, low_cpu_mem_usage=True)
            except:
                self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)

        self.model = self.model.cpu().eval()
        for param in self.model.parameters():
                param.requires_grad = False

    def rebuild_model(self):
        self.onnx_rebuilder.model_map = self.onnx_rebuilder.model_mapper.get_map(self.model.config)

        # load config
        ModelMapper.do_map(self.onnx_rebuilder, self.model.config, self.onnx_rebuilder.model_map['config'])
        if self.visual_length:
            ModelMapper.do_map(self.onnx_rebuilder.config.vision_config, self.model.config.vision_config, self.onnx_rebuilder.model_map['vision_config'])

        # rebuild config
        self.onnx_rebuilder.rebuild_config()

        # load modules
        ModelMapper.do_map(self.onnx_rebuilder, self.model, self.onnx_rebuilder.model_map['model'])

        # rebuild modules
        self.onnx_rebuilder.rebuild_modules()

    def export(self):
        self.onnx_rebuilder.export_config()
        self.onnx_rebuilder.export_tokenizer()

        self.onnx_rebuilder.export_embed()
        self.onnx_rebuilder.export_lm_head()

        if not self.lmhead_with_topk:
            self.onnx_rebuilder.export_greedy_head()
            self.onnx_rebuilder.export_penalty_sample_head()

        self.onnx_rebuilder.export_block()
        self.onnx_rebuilder.export_block_cache()

        if self.visual_length:
            self.onnx_rebuilder.export_processor()
            self.onnx_rebuilder.export_visual()

        if not self.not_compile:
            del self.model, self.onnx_rebuilder # need to delete at the same time
            self.bmodel_converter.compile()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='original weight, like ./Qwen2-7B-Instruct')
    parser.add_argument('-s', '--seq_length', type=int, required=True,
                        help="sequence length")
    parser.add_argument('-t', '--tpu_mlir_path', type=str,
                        help="tpu_mlir for compiling bmodel")
    parser.add_argument('-q', '--quantize', type=str,
                        choices=["bf16", "w8bf16", "w4bf16", "f16", "w8f16", "w4f16"],
                        help="quantize type for bmodel")
    parser.add_argument('-c', '--chip', type=str, default="bm1684x",
                        choices=["bm1684x", "bm1688", "cv186ah"],
                        help="chip type for bmodel")
    parser.add_argument('--num_device', type=int, default=1,
                        help="num device for bmodel")
    parser.add_argument('--not_compile', action='store_true',
                        help='only export onnx, not compile bmodel')
    parser.add_argument('--embedding_disk', action='store_true',
                        help='export embedding as bin file and inference by cpu')
    parser.add_argument('--out_dir', type=str, default='./tmp',
                        help='output onnx/bmodel path, default `./tmp`')
    parser.add_argument('--out_bmodel', type=str, default='',
                        help='combined bmodel name, default use original weight')
    parser.add_argument('--visual_length', type=int, default=0,
                        help="visual length in vision transformer for VLM")
    parser.add_argument('--max_workers', type=int, default=4,
                        help="max workers for compiling bmodel in multi-processing")
    parser.add_argument('--compile_mode', type=str, default="fast",
                        choices=["fast", "debug"],
                        help="compile with model_convert when use fast, compile with debug info when use debug")
    args = parser.parse_args()

    model_exporter = ModelExporter(args)
    model_exporter.export()
