import os
import sys
import json
import copy
import math
import torch
import onnx
from tqdm import tqdm
from typing import Optional, Tuple


from transformers import AutoTokenizer, AutoProcessor

def logging(message):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(message)
            return func(*args, **kwargs)
        return wrapper
    return decorator

class ModelMapper:
    def __init__(self):
        self.attrs = []
        self.mapper = dict()
        self.regist_models()

    def regist_models(self):
        self.default_map()
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
        self.regist_qwen2_vl()
        self.regist_qwen2_5_vl()
        self.regist_glm()
        self.regist_glm2()
        self.regist_phi()

    def default_map(self):
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
            'rope_theta': 'rope_theta',
            'vocab_size': 'vocab_size'
        }
        self.default_model = {
            'lm_': 'lm_head',
            'embed_': 'model.embed_tokens',
            'blocks_': 'model.layers',
            'final_layernorm_': 'model.norm',
            'visual_model': 'visual'
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
            'model': self.default_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }

    def regist(self, model_type, model_map):
        assert('config' in model_map and
               'decoder' in model_map and
               'attention' in model_map)
        self.mapper[model_type] = model_map

    def regist_qwen(self):
        qwen_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_hidden_layers',
                'rope_theta': 'rotary_emb_base',
                'vocab_size': 'vocab_size'
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
        
    def regist_qwen2_vl(self):
        qwen2_vl_map = {
            'config': self.default_config,
            'vision_config': {
                'embed_dim': 'embed_dim',
                'hidden_size': 'hidden_size'
            },
            'model': self.default_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }
        self.regist('qwen2_vl', qwen2_vl_map)

    def regist_qwen2_5_vl(self):
        qwen2_5_vl_map = {
            'config': self.default_config,
            'vision_config': {
                'embed_dim': 'hidden_size',
                'hidden_size': 'out_hidden_size'
            },
            'model': self.default_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }
        self.regist('qwen2_5_vl', qwen2_5_vl_map)

    def regist_glm(self):
        glm_map = {
            'config': {
                'hidden_size': 'hidden_size',
                'num_attention_heads': 'num_attention_heads',
                'num_hidden_layers': 'num_layers',
                'vocab_size': 'vocab_size'
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
                'vocab_size': 'vocab_size'
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
                'rotary_dim': 'rotary_dim',
                'vocab_size': 'vocab_size'
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

    def get_map(self, config):
        model_type = config.model_type
        if model_type in self.mapper:
            return self.mapper[model_type]
        return self.default_map

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

class OnnxRebuilder:
    def __init__(self,
				 onnx_dir: str,
                 model_path: str,
                 quantize: str,
				 seq_length: int,
                 model_type: str,
				 embedding_disk: bool,
                 config,
                 visual_length):
        self.onnx_model = None
        self.model_path = model_path
        self.quantize = quantize
        self.onnx_dir = onnx_dir
        self.seq_length = seq_length
        self.model_type = model_type
        self.embedding_disk = embedding_disk
        self.config = config
        self.visual_length = visual_length
        self.model_mapper = ModelMapper()

    def _replace_initializer(self, old_init, new_init):
        """
        Replaces the contents of an existing initializer with new data.

        Args:
            old_init (onnx.onnx_ml_pb2.TensorProto): The existing initializer to be replaced.
            new_init (onnx.onnx_ml_pb2.TensorProto): The new initializer with updated data.
        """
        old_init.CopyFrom(new_init)

    def rebuild_config(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.rope_theta is None:
            self.rope_theta = 10000.0
        self.head_dim = self.hidden_size // self.num_attention_heads

    def rebuild_modules(self):
        # Embedding
        if self.embed_.weight is self.lm_.weight:
            import copy
            embed_copy = copy.deepcopy(self.embed_)
            self.embed = Embedding(embed_copy, self.hidden_size)
        else:
            self.embed = Embedding(self.embed_, self.hidden_size)

        # Rotary
        self.rotary = Rotary(self)

        # Blocks
        self.blocks = []
        for block in self.blocks_.children():
            self.blocks.append(Decoder(block, self))

        # Lmhead
        self.lm = Lm(self)

        # Visual
        if self.visual_model is not None:
            self.visual = Visual.get_visual(self.model_type, self.visual_model, self)

    def rebuild_weights(self, torch_model, save_path):
        """
        Rebuild the weights from the PyTorch model into the ONNX model's initializers.
        Saves the updated ONNX model to the specified output path.
        """
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
                    new_init = onnx.numpy_helper.from_array(param, init_name)
                    # Replace the old initializer with the new one
                    self._replace_initializer(init, new_init)
                    matched += 1
                    break
        
        if matched != len(initializer_map) or matched != len(pytorch_params):
            raise ValueError("Match failed in onnx rebuild")

        # Save the modified model
        onnx.save(self.onnx_model, save_path)

    @logging("export_config ...")
    def export_config(self):
        config_dict = self.config.to_dict()
        with open(f'{self.onnx_dir}/../config.json', "w") as f:
            json.dump(config_dict, f, indent=4)
        return
    
    @logging("export_tokenizer ...")
    def export_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.save_pretrained(f'{self.onnx_dir}/../tokenizer')
        return

    @logging("export_processor ...")
    def export_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor.save_pretrained(f'{self.onnx_dir}/../processor')
        return

    @logging("export_embed ...")
    def export_embed(self):
        if not hasattr(self, 'embed') or not isinstance(self.embed.embed, torch.nn.Embedding):
            return
        if not self.embedding_disk:
            embedding_file = f'{self.onnx_dir}/embedding.pt'
            if os.path.exists(embedding_file):
                print(f"{embedding_file} already exists. Skipping export.")
                return
            input_ids = torch.tensor([range(self.seq_length)], dtype=torch.int32)
            module = torch.jit.trace(self.embed.forward, input_ids)
            torch.jit.save(module, embedding_file)
        else:
            embedding_file = f'{self.onnx_dir}/../embedding.bin'
            if os.path.exists(embedding_file):
                print(f"{embedding_file} already exists. Skipping export.")
                return
            import ctypes
            if 'bf16' in self.quantize:
                tensor_data = self.embed.embed.weight.data.to(torch.bfloat16)
            elif 'f16' in self.quantize:
            	tensor_data = self.embed.embed.weight.data.to(torch.float16)
            else:
                raise NotImplementedError("Not support now")
            data_ptr = tensor_data.untyped_storage().data_ptr()
            buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
            with open(embedding_file, 'wb') as f:
                f.write(buffer)

    @logging("export_block ...")
    def export_block(self):
        hidden_states = torch.randn((1, self.seq_length, self.hidden_size), dtype=torch.float)
        position_ids = torch.tensor([range(self.seq_length)], dtype=torch.long)
        attention_mask = torch.randn((1, 1, self.seq_length, self.seq_length), dtype=torch.float)

        for i in tqdm(range(self.num_hidden_layers)):
            model = self.blocks[i]
            onnx_path = f'{self.onnx_dir}/block_{i}.onnx'
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
                self.onnx_model = onnx.load(onnx_path)
            else:
                self.rebuild_weights(
                    torch_model=model,
                    save_path=onnx_path
                )

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

        for i in tqdm(range(len(self.blocks))):
            model = self.blocks[i]
            onnx_path = f'{self.onnx_dir}/block_cache_{i}.onnx'
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
                self.onnx_model = onnx.load(onnx_path)
            else:
                self.rebuild_weights(
                    torch_model=model,
                    save_path=onnx_path
                )

    @logging("export_lm_head ...")
    def export_lm_head(self):
        lmhead_file = f'{self.onnx_dir}/lm_head.pt'
        if os.path.exists(lmhead_file):
            print(f"{lmhead_file} already exists. Skipping export.")
            return
        model = self.lm
        hidden_states = torch.randn((1, 1, self.hidden_size), dtype=torch.float)
        module = torch.jit.trace(model.forward, hidden_states)
        torch.jit.save(module, lmhead_file)

    @logging("export_greedy_head ...")
    def export_greedy_head(self):
        onnx_path = f'{self.onnx_dir}/greedy_head.onnx'
        if os.path.exists(onnx_path):
            print(f"{onnx_path} already exists. Skipping export.")
            return

        model = GreedyHead()
        m_logits = torch.randn(1, self.vocab_size)
        torch.onnx.export(
            model, (m_logits),
            onnx_path,
            verbose=False,
            input_names=['m_logits'],
            output_names=['token'],
            do_constant_folding=True,
            opset_version=15)

    @logging("export_penalty_sample_head ...")
    def export_penalty_sample_head(self):
        onnx_path = f'{self.onnx_dir}/penalty_sample_head.onnx'
        if os.path.exists(onnx_path):
            print(f"{onnx_path} already exists. Skipping export.")
            return

        model = PenaltySampleHead()
        m_logits = torch.randn(1, self.vocab_size)
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

    @logging("export_visual...")
    def export_visual(self):
        dir_path = os.path.join(self.onnx_dir, "vit")
        os.makedirs(dir_path, exist_ok=True)

        onnx_path = os.path.join(dir_path, "vit.onnx")
        if os.path.exists(onnx_path):
            print(f"{onnx_path} already exists. Skipping export.")
            return
        return self.visual.export(onnx_path)

# some wrapper class for export
class Embedding(torch.nn.Module):
    def __init__(self, embed, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = embed

    def forward(self, input_ids):
        return self.embed(input_ids).view(1, -1, self.hidden_size)

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
                split_sizes = [
                    self.num_heads * self.head_dim,           # q_size
                    self.num_key_value_heads * self.head_dim, # k_size
                    self.num_key_value_heads * self.head_dim  # v_size
                ]

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
                else:
                    self.q_proj.bias.data = torch.zeros(split_sizes[0])
                    self.k_proj.bias.data = torch.zeros(split_sizes[1])
                    self.v_proj.bias.data = torch.zeros(split_sizes[2])
            self.q_proj.weight.requires_grad = False
            self.k_proj.weight.requires_grad = False
            self.v_proj.weight.requires_grad = False
            self.q_proj.bias.requires_grad = False
            self.k_proj.bias.requires_grad = False
            self.v_proj.bias.requires_grad = False
            del self.qkv_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
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
        if rotary_pos_emb is None:
            cos, sin = self.rotary.cos[position_ids], self.rotary.sin[position_ids]
        else:
            cos, sin = rotary_pos_emb
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
        attn_weights = torch.matmul(query_states.transpose(1, 2), key_states.transpose(1, 2).transpose(2, 3)) / (self.head_dim ** 0.5)
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
    def __init__(self, config):
        super().__init__()
        self.seq_length = config.seq_length
        self.rope_theta = config.rope_theta
        self.rotary_dim = config.head_dim
        self.model_type = config.model_type
        if hasattr(config, 'rotary_dim'):
            self.rotary_dim = config.rotary_dim
        if self.model_type == 'chatglm':
            self.rotary_dim = config.head_dim // 2

        self.cos, self.sin = self.init_rotary_pos_emb(self.seq_length)
        self.cos = self.cos.squeeze(0)
        self.sin = self.sin.squeeze(0)

    def init_rotary_pos_emb(self, seq_length):
        position_ids = torch.tensor([range(seq_length)], dtype=torch.long)
        theta = 1.0 / (self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        if self.model_type != 'chatglm2':
            rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        return rotary_pos_emb

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
        rotary_pos_emb: Optional[Tuple[torch.Tensor]] = None,
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
            rotary_pos_emb=rotary_pos_emb
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
    def __init__(self, config):
        super().__init__()
        self.final_layernorm = config.final_layernorm_
        self.lm = config.lm_
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

class Visual(torch.nn.Module):
    def __init__(self, visual_model, base):
        super().__init__()
        self.model_type = base.model_type
        self.visual_model = visual_model.eval()
        self.embed_ = base.embed
        self.config = base.config
        self.visual_config = self.config.vision_config

        if not hasattr(base, 'visual_length') or base.visual_length <= 0:
            raise ValueError("Please provide a int number above zero for --visual_length, when converting Vision Language Model.")

        self.visual_length = base.visual_length
        self.rope_theta = base.rope_theta

        self.hidden_size = self.visual_config.embed_dim
        self.num_attention_heads = self.visual_config.num_heads
        self.num_key_value_heads = self.visual_config.num_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rotary = VisionRotary(self)
        self.model_map = self.get_model_map()
        self.load()

    @staticmethod
    def get_visual(model_type, visual_model, base):
        visual_models = {
            'qwen2_vl': Qwen2Visual,
            'qwen2_5_vl': Qwen2Visual,
        }
        if model_type in visual_models:
            return visual_models[model_type](visual_model, base)
        else:
            raise NotImplementedError("Not support now")
        return None

    def get_model_map(self):
        if self.model_type == "qwen2_vl" or self.model_type == "qwen2_5_vl":
            model_map = {
                'decoder': {
                    'self_attn': 'attn',
                    'mlp': 'mlp',
                    'input_layernorm': 'norm1',
                    'post_attention_layernorm': 'norm2'
                },
                'attention': {
                    'qkv_proj': 'qkv',
                    'o_proj': 'proj'
                }
            }
        return model_map

    def export(self, onnx_path):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def forward(self, images):
        raise NotImplementedError

class VisionRotary(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual_length = config.visual_length
        self.rope_theta = config.rope_theta
        self.rotary_dim = config.head_dim
        self.model_type = config.model_type
        if hasattr(config, 'rotary_dim'):
            self.rotary_dim = config.rotary_dim
        if self.model_type == 'chatglm':
            self.rotary_dim = config.head_dim // 2
        self.inv_freq = config.visual_model.rotary_pos_emb.inv_freq
        self.cos, self.sin = self.init_rotary_pos_emb(self.visual_length)

    def init_rotary_pos_emb(self, visual_length):
        if self.model_type == "qwen2_vl" or self.model_type == "qwen2_5_vl":
            seq = torch.arange(visual_length)
            freqs = torch.outer(seq, self.inv_freq)
            self.cos = freqs.cos()
            self.sin = freqs.sin()
        return (self.cos, self.sin)

    def llama_rotary_pos(self, x, cos, sin):
        x = (x * cos) + (rotate_half(x) * sin)
        return x

    def apply_rotary_pos(self, x, cos, sin):
        if self.model_type == 'chatglm':
            return self.chatglm_rotary_pos(x, cos, sin)
        if self.model_type == 'chatglm2':
            return self.chatglm2_rotary_pos(x, cos, sin)
        if self.model_type == 'phi-msft':
            return self.phi_rotary_pos(x, cos, sin)
        return self.llama_rotary_pos(x, cos, sin)

class Qwen2Visual(Visual):
    def __init__(self, visual_model, base):
        super().__init__(visual_model, base)

    def load(self):
        # load model
        self.hidden_size = self.visual_config.embed_dim
        self.num_attention_heads = self.visual_config.num_heads
        self.num_key_value_heads = self.visual_config.num_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rotary_dim = self.head_dim // 2
        self.patch_embed = self.visual_model.patch_embed
        self.blocks = []
        for block in self.visual_model.blocks.children():
            self.blocks.append(Decoder(block, self))
        self.merger = self.visual_model.merger
        self.spatial_merge_size = self.visual_config.spatial_merge_size
        self.max_pixels = self.visual_length * self.spatial_merge_size * self.spatial_merge_size
        self.mlp_ratio = self.merger.mlp[0].weight.shape[0] // self.hidden_size

        self.in_channels = self.patch_embed.proj.in_channels
        self.conv_dim = math.prod(self.patch_embed.proj.kernel_size) * self.in_channels

    def forward(self, flatten_patches, position_ids, attention_mask):
        self.cos = self.rotary.cos[position_ids].flatten(1).unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)
        self.sin = self.rotary.sin[position_ids].flatten(1).unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)

        hidden_states = self.patch_embed(flatten_patches)
        for blk in self.blocks:
            hidden_states, _ = blk(hidden_states, rotary_pos_emb=(self.cos, self.sin), attention_mask=attention_mask)
        image_embeds = self.merger.mlp(self.merger.ln_q(hidden_states).view(1, -1, self.hidden_size * self.mlp_ratio))
        return image_embeds

    def export(self, onnx_path):
        patch = torch.randn(self.max_pixels, self.conv_dim).to(dtype=torch.float32)
        position_ids = torch.randn(self.max_pixels, 2).to(dtype=torch.int32)
        attention_mask = torch.zeros([1, 1, self.max_pixels, self.max_pixels], dtype=torch.float32)

        torch.onnx.export(
            self, (patch, position_ids, attention_mask),
            onnx_path,
            verbose=False,
            input_names=['input_states', 'position_ids', 'attention_mask'],
            output_names=['hidden_states'],
            do_constant_folding=True,
            opset_version=15
        )
        return
