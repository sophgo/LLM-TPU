import sys, os
import argparse
import torch
from modelscope import AutoModel, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from support.preprocess import process_image

class VisionTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample_ratio = model.downsample_ratio

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, pixel_values):
        # [n, 3, 448, 448] -> [n, 1024, vit_hidden_size]
        vit_embeds = vit(
            pixel_values=pixel_values, 
            output_hidden_states=False,
            return_dict=True).last_hidden_state
        vit_embeds = vit_embeds[:, 1:, :]

        # [n, 1024, vit_hidden_size] -> [n, 256, 4*vit_hidden_size] 
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        # [n, 256, 4*vit_hidden_size] -> [n, 256, llm_hidden_size]
        vit_embeds = mlp(vit_embeds).reshape(-1, HIDDEN_SIZE)

        return vit_embeds

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return llm.model.embed_tokens(input_ids)

class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer = llm.model.layers[layer_id]
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
        self.layer = llm.model.layers[layer_id]
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
        hidden_states = llm.model.norm(hidden_states)
        m_logits = llm.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token

def convert_vision_transformer():
    model = VisionTransformer()
    images = torch.randn(1, 3, 448, 448, dtype=dtype)
    torch.onnx.export(
        model, images,
        f"{folder}/vit/vit.onnx",
        input_names=["pixel_values"],
        output_names=["vit_embeds"],
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
    convert_vision_transformer()
    convert_embedding()
    convert_lm_head()
    for i in range(NUM_LAYERS):
        print(f'export block {i} ...')
        convert_block(i)
        convert_block_cache(i)
 
def test_net_with_mask(image_path):
    question = 'describe this image in detail.'
    pixel_values = process_image(image_path)
    print(f"Loading {image_path}")
    print(f"Question: {question}")
    system_message = '你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'
    query = f'<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n'
    IMG_START_TOKEN='<img>'
    IMG_END_TOKEN='</img>'
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
    IMG_TOKEN_ID = torch.tensor(tokenizer.encode('<IMG_CONTEXT>'))
    num_patches = pixel_values.shape[0]
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * NUM_IMAGE_TOKEN * num_patches + IMG_END_TOKEN
    VISUAL_LEN = NUM_IMAGE_TOKEN * num_patches
    query = query.replace('<image>', image_tokens, 1)

    vit = VisionTransformer()
    embed = Embedding()
    lm = LmHead()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]

    ids = tokenizer(query)['input_ids']
    token_len = len(ids)
    ids = ids + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH)
    vit_offset = torch.where(input_ids==IMG_TOKEN_ID)[0][0].item()

    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
    vit_embeds = []
    for i in range(pixel_values.shape[0]):
        vit_embeds.append(vit(pixel_values[i:i+1]))
    vit_embeds = torch.cat(vit_embeds, dim=0)
    out[0, vit_offset:vit_offset + VISUAL_LEN] = vit_embeds

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
    print("\nAnswer: ")
    while int(token) != tokenizer.eos_token_id:
        out_ids.append(int(token))
        word = tokenizer.decode([int(token)])
        print(word, end='', flush=True)
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

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

    vit = model.vision_model
    llm = model.language_model
    mlp = model.mlp1

    SEQ_LENGTH = args.seq_length
    NUM_LAYERS = model.config.llm_config.num_hidden_layers
    HIDDEN_SIZE = model.config.llm_config.hidden_size
    NUM_HEADS = model.config.llm_config.num_attention_heads
    NUM_KEY_VALUE_HEADS = model.config.llm_config.num_key_value_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    NUM_IMAGE_TOKEN = model.num_image_token

    for param in model.parameters():
        param.requires_grad = False

    if args.test:
        image_path = '../python_demo/test.jpg' if args.image_path is None else args.image_path
        with torch.no_grad():
            test_net_with_mask(image_path)
    else:
        folder = 'tmp/onnx'
        os.makedirs(folder+'/vit', exist_ok=True)
        convert()