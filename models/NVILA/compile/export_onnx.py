import torch
import sys, os
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import SiglipImageProcessor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from support.llava.base_projector import MultimodalProjector, MultimodalProjectorConfig
from support.llava.modeling_siglip import SiglipVisionModel
from support.llava.configuration_llava import LlavaConfig

from einops import rearrange

class Vision_Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = vision_tower

    def forward(self, images):
        image_forward_out = self.vision_tower(
            images,
            output_hidden_states=True,
        )
        image_feature = image_forward_out.hidden_states[-2]

        return image_feature

class Dynamic_S2_Merge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_projector = mm_projector
        self.scales = list(map(int, config.s2_scales.split(",")))
        self.resize_output_to_scale_idx = config.s2_resize_output_to_scale_idx
        self.end_token_embeds = llama.embed_tokens(torch.tensor([198]))

    @staticmethod
    def merge_chessboard(x, num_split_h, num_split_w, N):
        x_merge = rearrange(x, "(nh nw) (h w) c-> 1 c (nh h) (nw w)", nh=num_split_h, nw=num_split_w, h=N, w=N)
        return x_merge

    @staticmethod
    def split_chessboard(x, num_split_h, num_split_w):
        x_split = rearrange(x, "1 c (nh h) (nw w) -> (nh nw) c h w", nh=num_split_h, nw=num_split_w)
        return x_split

    def forward(self, image_feature):
        # bh, bw = block_size
        # bn = bh * bw + 5 = image_feature.shape[0]
        #  cur_features_each_scale[448,896,1344]:
        #  [1, 1024, 1152] -> [1, 1152, 32, 32]
        #  [4, 1024, 1152] -> [1, 1152, 64, 64]
        #  [bn-5, 1024, 1152] -> [1, 1152, 32*bh, 32*bw]
        # scale0_feature = self.merge_chessboard(
        #     image_feature[0:1], num_split_h=1, num_split_w=1, N=32)
        # scale1_feature = self.merge_chessboard(
        #     image_feature[1:5], num_split_h=2, num_split_w=2, N=32)
        # scale2_feature = self.merge_chessboard(
        #     image_feature[5:], block_size[0], block_size[1], N=32)

        # resize and concat features from different scales
        # [1, 1152*3, 32*bh, 32*bw]
        # output_size = torch.Size([32*block_size[0], 32*block_size[1]])
        # image_feature = torch.cat(
        #     [
        #         torch.nn.functional.interpolate(scale0_feature, size=output_size),
        #         torch.nn.functional.interpolate(scale1_feature, size=output_size),
        #         scale2_feature
        #     ],
        #     dim=1,
        # )
        # [1, 1152*3, 32*bh, 32*bw] -> [bn-5, 3456, 32, 32]
        # image_feature = self.split_chessboard(
        #     image_feature, block_size[0], block_size[1])
        # [bn-5, 3456, 32, 32] -> [bn-5, 1024, 3456]
        image_feature = rearrange(image_feature, "b c h w -> b (h w) c")
        # [bn-5, 1024, 3456] -> [bn-5, 256, 3584]
        image_feature = self.mm_projector(image_feature)
        # [bn-5, 256, 3584] -> [1, 3584, 16*bh, 16*bw]
        # image_feature = self.merge_chessboard(
        #     image_feature, block_size[0], block_size[1], N=16)
        # # [1, 3584, 16*bh, 16*bw] -> [(bn-5)*256, 3584]
        # image_feature = rearrange(image_feature, "1 c h w -> (h w) c")
        # # [(bn-5)*256, 3584] -> [(bn-5)*256+1, 3584]
        # image_feature = torch.cat([image_feature, self.end_token_embeds], dim=0)
        return image_feature

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return llama.embed_tokens(input_ids)

class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer = llama.layers[layer_id]
        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
        position_ids = torch.tensor([range(SEQ_LENGTH)],dtype=torch.long).to(device)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            use_cache=True,
                                            position_embeddings=(self.cos, self.sin))
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v
    
class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer = llama.layers[layer_id]
        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
        position_ids = torch.tensor([range(SEQ_LENGTH)],dtype=torch.long).to(device)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True,
                                            position_embeddings=(self.cos, self.sin))
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v

class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = llm_model.model.norm(hidden_states)
        m_logits = llm_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token

def convert_vision_embedding():
    model = Vision_Embedding()
    images = torch.randn(1, 3, 448, 448, dtype=dtype)
    torch.onnx.export(
        model, images,
        f"{folder}/vision_embedding.onnx",
        input_names=["images"],
        output_names=["image_feature"],
        do_constant_folding=True,
        opset_version=15
    )

def convert_dynamic_s2_merge():
    model = Dynamic_S2_Merge()
    image_feature = torch.randn(20, 3456, 32, 32, dtype=dtype)
    torch.onnx.export(
        model, (image_feature),
        f"{folder}/merge.onnx",
        input_names=["image_feature"],
        output_names=["media_embeds"],
        do_constant_folding=True,
        opset_version=15
    )

def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)]).to(device=device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f"{folder}/embedding.pt")

def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE), dtype=dtype).to(device=device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device=device)
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype=torch.float32).triu(diagonal=1).to(device=device)

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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE), dtype=dtype).to(device=device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device=device)
    attention_mask = -1000 * torch.ones((1, 1, 1, SEQ_LENGTH + 1), dtype=torch.float32).triu(diagonal=1).to(device=device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM), dtype=dtype).to(device=device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM), dtype=dtype).to(device=device)

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

def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(dtype).to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f"{folder}/lm_head.pt")

def convert():
    convert_vision_embedding()
    convert_dynamic_s2_merge()
    convert_embedding()
    convert_lm_head()
    for i in range(NUM_LAYERS):
        print(f'export block {i} ...')
        convert_block(i)
        convert_block_cache(i)

def test_net_with_mask(image_path):
    from PIL import Image
    from support.preprocess import process_image

    prompt = 'introduce this image'
    image = Image.open(image_path).convert('RGB')
    print(f"Loading {image_path}")
    print(f"Question: {prompt}")
    text = f'<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n<image>{prompt}<|im_end|>\n<|im_start|>assistant\n'

    vit = Vision_Embedding()
    merge = Dynamic_S2_Merge()
    embed = Embedding()
    lm = LmHead()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]

    # images [h, w] -> [num_blocks, 3, 448, 448]
    # num_blocks = 1 + 4 + block_size[0] * block_size[1]
    images, block_size = process_image(image, image_processor)
    VISUAL_LEN = block_size[0] * block_size[1] * 256 + 1
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    token_len = len(ids)
    ids = ids.tolist() + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH).to(device)
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)

    # image_feature [num_blocks, 3, 448, 448] -> [num_blocks, 1024, 1152]
    # image_feature [num_blocks, 1024, 1152] -> [(num_blocks-5)*256+1, 3584]
    image_feature = vit(images)
    scale0_feature = merge.merge_chessboard(
        image_feature[0:1], num_split_h=1, num_split_w=1, N=32)
    scale1_feature = merge.merge_chessboard(
        image_feature[1:5], num_split_h=2, num_split_w=2, N=32)
    scale2_feature = merge.merge_chessboard(
        image_feature[5:], block_size[0], block_size[1], N=32)
    output_size = torch.Size([32*block_size[0], 32*block_size[1]])
    image_feature = torch.cat(
        [
            torch.nn.functional.interpolate(
                scale0_feature, size=(32*block_size[0], 32*block_size[1])),
            torch.nn.functional.interpolate(
                scale1_feature, size=(32*block_size[0], 32*block_size[1])),
            scale2_feature
        ],
        dim=1,
    )
    image_feature = merge.split_chessboard(image_feature, block_size[0], block_size[1])
    image_feature = merge(image_feature)
    image_feature = merge.merge_chessboard(
        image_feature, block_size[0], block_size[1], N=16)
    image_feature = rearrange(image_feature, "1 c h w -> (h w) c")
    image_feature = torch.cat([image_feature, merge.end_token_embeds], dim=0)

    out[:, 13 + VISUAL_LEN:VISUAL_LEN + token_len - 1] = out[:, 14:token_len]
    out[:, 13:13 + VISUAL_LEN] = image_feature.unsqueeze(0)
    token_len += VISUAL_LEN - 1

    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids]).to(device)
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH)

    k_cache = []
    v_cache = []
    for i in range(NUM_LAYERS):
        out, k, v = blocks[i](out.to(dtype), position_ids,
                              attention_mask.to(dtype))
        k[:, :, token_len:] = 0
        v[:, :, token_len:] = 0
        k_cache.append(k)
        v_cache.append(v)
    out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)
    token = lm(out.to(dtype)).view(1)
    out_ids = []
    while int(token) != tokenizer.eos_token_id:
        out_ids.append(int(token))
        word = tokenizer.decode([int(token)])
        print(word, end='')
        token_len += 1
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float()
        attention_mask[:, :, :, token_len-1:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out.to(dtype), position_ids,
                                     attention_mask.to(dtype),
                                     k_cache[i].to(dtype), v_cache[i].to(dtype))
            k_cache[i][:, token_len-1:token_len] = k
            v_cache[i][:, token_len-1:token_len] = v
        token = lm(out.to(dtype)).view(1)
    print("\noutput_ids:{}".format(out_ids))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
    parser.add_argument('-s', '--seq_length', type=int, default=4096, help="sequence length")
    parser.add_argument('-i', '--image_path', type=str, help='path to image for test')
    parser.add_argument('-t', '--test', action='store_true', help="sequence length")
    args = parser.parse_args()

    model_path = args.model_path
    device = torch.device("cpu")
    dtype = torch.float

    folder = 'tmp/onnx'
    os.makedirs(folder, exist_ok=True)

    llm_path = os.path.join(model_path, "llm")
    vision_path = os.path.join(model_path, "vision_tower")
    proj_path = os.path.join(model_path, "mm_projector")
    config = LlavaConfig.from_pretrained(model_path)
    llm_cfg = AutoConfig.from_pretrained(llm_path)

    llm_model = AutoModelForCausalLM.from_pretrained(
      llm_path, config=llm_cfg, torch_dtype=dtype, attn_implementation='eager').eval()
    llm_model.resize_token_embeddings(llm_cfg.vocab_size + 3)
    vision_tower = SiglipVisionModel.from_pretrained(vision_path, torch_dtype=dtype, attn_implementation='eager').eval()
    mm_projector = MultimodalProjector.from_pretrained(proj_path, config, torch_dtype=dtype).eval()

    llama = llm_model.model
    NUM_LAYERS = llm_cfg.num_hidden_layers
    HIDDEN_SIZE = llm_cfg.hidden_size
    SEQ_LENGTH = args.seq_length
    NUM_HEADS = llm_cfg.num_attention_heads
    NUM_KEY_VALUE_HEADS = llm_cfg.num_key_value_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS

    for param in llm_model.parameters():
        param.requires_grad = False

    if args.test:
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        tokenizer.stop_tokens = ['<|im_end|>']
        tokenizer.stop_token_ids = [151645]
        tokenizer.add_tokens(["<vila/sentinel>"], special_tokens=True)
        tokenizer.sentinel_token = "<vila/sentinel>"
        tokenizer.sentinel_token_id = [151648]
        tokenizer.media_tokens = {"image": "<image>","video": "<vila/video>",}
        tokenizer.media_token_ids = {}
        for name, token in tokenizer.media_tokens.items():
            tokenizer.add_tokens([token], special_tokens=True)
            tokenizer.media_token_ids[name] = tokenizer.convert_tokens_to_ids(token)
        image_processor = SiglipImageProcessor.from_pretrained(vision_path)
        image_path = '../python_demo/test.png' if args.image_path is None else args.image_path
        with torch.no_grad():
            test_net_with_mask(image_path)
    else:
        convert()