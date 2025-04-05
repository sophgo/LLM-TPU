from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
import torch
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image

torch.set_grad_enabled(False)

folder = f"./tmp/onnx"

pretrained = "/workspace/models/DriveMM/"
model_name = 'llama'  
device = torch.device('cpu')
llava_model_args = {
        "multimodal": True,
        "attn_implementation": "eager",  # Using eager for CPU compatibility
        "torch_dtype": torch.float32,
        
    }
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device, **llava_model_args)
model.eval()

for param in model.parameters():
    param.requires_grad = False
    
    
output_dir = "./exported_tokenizer"
os.makedirs(output_dir, exist_ok=True)

# 保存 tokenizer 及相关配置文件
tokenizer.save_pretrained(output_dir)


config = model.config
transformer = model.model
layers = transformer.layers

SEQ_LENGTH = 2048
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.vocab_size

class ImageEncoder(torch.nn.Module):
    def __init__(self, vision_tower, mm_projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector

    def forward(self, images):
        images = images.to(dtype=torch.float32)
        self.vision_tower.to(device).to(torch.float32)
        self.mm_projector.to(device).to(torch.float32)
        with torch.autocast(device_type='cpu', enabled=False):
            image_features = self.vision_tower(images)

        image_features = torch.cat((self.mm_projector(image_features), model.model.image_newline[None, None, :]), dim=1)
        return image_features

def convert_image_encoder():
    device = torch.device('cpu')
    vision_tower = model.get_model().vision_tower.to(device).to(torch.float32)
    mm_projector = model.get_model().mm_projector.to(device).to(torch.float32)
    
    # 检查所有参数类型
    for name, param in vision_tower.named_parameters():
        assert param.dtype == torch.float32, f"{name} 类型错误: {param.dtype}"
    
    image_encoder = ImageEncoder(vision_tower, mm_projector).eval()
    dummy_images = torch.randn(1, 3, 384, 384, dtype=torch.float32, device=device)
    
    # 导出时使用 opset 18
    torch.onnx.export(
        image_encoder,
        dummy_images,
        f'{folder}/image_encoder.onnx',
        input_names=['pixel_values'],
        output_names=['image_features'],
        opset_version=18  # 关键修改
    )

       
class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return transformer.embed_tokens(input_ids)
    
class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = model.lm_head(hidden_states)
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
    
def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE))
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype=torch.float32).triu(diagonal=1)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE))
    position_ids = torch.tensor([range(1)], dtype=torch.long)
    attention_mask = -1000 * torch.ones((1, 1, 1, SEQ_LENGTH + 1), dtype=torch.float32).triu(diagonal=1)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM))
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM))

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'{folder}/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'position_ids', 'attention_mask', 'history_k',
            'history_v'
        ],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)

def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')


def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(torch.float32).to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')

def convert_greedy_head():   
    model = GreedyHead()
    m_logits = torch.randn(1, VOCAB_SIZE)

    torch.onnx.export(
        model, (m_logits),
        f'{folder}/greedy_head.onnx',
        verbose=False,
        input_names=['m_logits'],
        output_names=['token'],
        do_constant_folding=True,
        opset_version=15)


def convert_penalty_sample_head():   
    model = PenaltySampleHead()
    m_logits = torch.randn(1, VOCAB_SIZE)
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    top_p = torch.tensor([0.8])
    temperature = torch.tensor([0.98])
    penalty = torch.tensor([0.98])

    torch.onnx.export(
        model, (m_logits, input_ids, top_p, temperature, penalty),
        f'{folder}/penalty_sample_head.onnx',
        verbose=False,
        input_names=[
            'm_logits', 'input_ids', 'top_p', 'temperature',
            'penalty'
        ],
        output_names=['probs', 'token'],
        do_constant_folding=True,
        opset_version=15)

def export_vision_tower():
    device = torch.device('cpu')
    vision_tower = model.get_model().vision_tower.to(device).to(torch.float32)
    
    # 修复动态形状检查问题
    for layer in vision_tower.modules():
        if hasattr(layer, 'num_heads'):
            # 修改注意力层的动态判断逻辑
            def _forward_hook(module, input, output):
                # 替换原始代码中的 if attn_weights.size() != ... 判断
                return output
            layer.register_forward_hook(_forward_hook)
    
    vision_model = VisionTowerWrapper(vision_tower).eval()
    dummy_images = torch.randn(1, 3, 384, 384, dtype=torch.float32, device=device)
    
    torch.onnx.export(
        vision_model,
        dummy_images,
        "vision_tower.onnx",
        input_names=["pixel_values"],
        output_names=["vision_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
            "vision_features": {0: "batch_size"}
        },
        opset_version=18
    )

def export_projector():
    device = torch.device('cpu')
    mm_projector = model.get_model().mm_projector.to(device).to(torch.float32)
    
    # 生成中间输入样本（需与 Vision Tower 输出形状匹配）
    dummy_vision_features = torch.randn(1, 729, 1024, dtype=torch.float32, device=device)  # 假设 Vision Tower 输出形状为 [1, 729, 1024]
    
    projector_model = ProjectorWrapper(mm_projector).eval()
    
    torch.onnx.export(
        projector_model,
        dummy_vision_features,
        "projector.onnx",
        input_names=["vision_features"],
        output_names=["image_features"],
        dynamic_axes={
            "vision_features": {0: "batch_size"},
            "image_features": {0: "batch_size"}
        },
        opset_version=18
    )




from functools import partial, reduce
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)

size=(384,384)
resample=PILImageResampling.BICUBIC
data_format=ChannelDimension.FIRST
rescale_factor=0.00392156862745098
image_mean=(0.5, 0.5, 0.5)
image_std=(0.5, 0.5, 0.5)
def preprocess(images, return_tensors="pt"):
    if isinstance(images, Image.Image):
        images = [images]
    else:
        # to adapt video data
        images = [to_numpy_array(image) for image in images]
        assert isinstance(images, list)

    transforms = [
        convert_to_rgb,
        to_numpy_array,
        partial(resize, size=size, resample=resample, data_format=data_format),
        partial(rescale, scale=rescale_factor, data_format=data_format),
        partial(normalize, mean=image_mean, std=image_std, data_format=data_format),
        partial(to_channel_dimension_format, channel_dim=data_format, input_channel_dim=data_format),
    ]

    images = reduce(lambda x, f: [*map(f, x)], transforms, images)
    return images


import copy

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids
    
from llava.conversation import conv_templates, SeparatorStyle

def test_net_with_mask():
    urls = ['../python_demo/codalm3.png']
    question = "<image>\nThere is an image of traffic captured from the front view of the ego vehicle. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene."
    modalities=['image']
    images = [Image.open(str(url)).convert("RGB") for url in urls]
    image_tensors = preprocess(images)
    image_tensors = [torch.tensor(image, dtype=torch.float32) for image in image_tensors]
    image_sizes = [image.size for image in images]
    
    conv_template = "llava_llama_3"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    
    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        modalities=modalities,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(text_outputs[0])


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

test_net_with_mask()
exit()


print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block_cache(i)
    convert_block(i)

print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()
convert_greedy_head()
convert_penalty_sample_head()

print(f'Convert image encode')
convert_image_encoder()


print("Done")

