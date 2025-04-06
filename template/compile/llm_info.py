from typing import List
from enum import Enum


class ModelConfig:

    def __init__(self,
                 num_attention_heads: str = 'num_attention_heads',
                 num_hidden_layers: str = 'num_hidden_layers',
                 num_key_value_heads: str = 'num_key_value_heads',
                 hidden_size: str = 'hidden_size',
                 vocab_size: str = 'vocab_size',
                 intermediate_size: str = 'intermediate_size',
                 rope_theta: str = "rope_theta"):
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta


class WeightType(Enum):
    NORMAL = "NORMAL"
    MM_WEIGHT = "MM_WEIGHT"
    MM_BIAS = "MM_BIAS"
    ROTARY_COS = "ROTARY_COS"
    ROTARY_SIN = "ROTARY_SIN"


class WeightInfo:

    def __init__(self, name: str, weight_type: WeightType):
        self.name = name
        self.type = weight_type


class ModelWeights:

    def __init__(self, layers: str, embed: WeightInfo, blocks: List[WeightInfo], norm: WeightInfo,
                 lm_head: WeightInfo):
        self.layers = layers
        self.embed = embed
        self.blocks = blocks
        self.norm = norm
        self.lm_head = lm_head


class ModelInfo:

    def __init__(self, config: ModelConfig, weights: ModelWeights):
        self.config = config
        self.weights = weights


QWEN2_INFO = ModelInfo(
    ModelConfig(),
    ModelWeights(
        embed=WeightInfo("model.embed_tokens.weight", WeightType.NORMAL),
        layers="model.layers",
        blocks=[
            WeightInfo("input_layernorm.weight", WeightType.NORMAL),
            WeightInfo("self_attn.q_proj", WeightType.MM_WEIGHT),
            WeightInfo("self_attn.q_proj", WeightType.MM_BIAS),
            WeightInfo("self_attn.k_proj", WeightType.MM_WEIGHT),
            WeightInfo("self_attn.k_proj", WeightType.MM_BIAS),
            WeightInfo("self_attn.v_proj", WeightType.MM_WEIGHT),
            WeightInfo("self_attn.v_proj", WeightType.MM_BIAS),
            WeightInfo("", WeightType.ROTARY_COS),
            WeightInfo("", WeightType.ROTARY_SIN),
            WeightInfo("self_attn.o_proj", WeightType.MM_WEIGHT),
            WeightInfo("post_attention_layernorm.weight", WeightType.NORMAL),
            WeightInfo("mlp.gate_proj", WeightType.MM_WEIGHT),
            WeightInfo("mlp.up_proj", WeightType.MM_WEIGHT),
            WeightInfo("mlp.down_proj", WeightType.MM_WEIGHT),
        ],
        norm=WeightInfo("model.norm.weight", WeightType.NORMAL),
        lm_head=WeightInfo("lm_head", WeightType.MM_WEIGHT),
    ))
