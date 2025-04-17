import os, sys, re
import argparse
from typing import List
import torch
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER


parser = argparse.ArgumentParser(description='export rwkv')
parser.add_argument('-m', '--model_path', type=str, help='path to rwkv model file')
parser.add_argument('-c', '--chunk_len', type=int, default=32, help="chunk len for forward seq")
parser.add_argument('-s', '--state_path', type=str, default=None, help="path to rwkv state file")
parser.add_argument('-t', '--test', action='store_true', help="test original net by cpu")

folder = r'./tmp'
os.makedirs(folder, exist_ok=True)
os.makedirs(folder+'/one', exist_ok=True)
os.makedirs(folder+'/seq', exist_ok=True)

args = parser.parse_args()
DTYPE = torch.float
DEVICE = "cpu"
STATE = args.state_path
CHUNK_LEN = args.chunk_len

def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

    xx = torch.nn.functional.group_norm(xx.view(1,H*N,1), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, x, state, v_first

def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
    for t in range(T):
        r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
        vk = v_.view(H,N,1) @ k_.view(H,1,N)
        ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
        state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
        xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

    xx = torch.nn.functional.group_norm(xx.view(T,H*N,1), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return (xx * g) @ O_, x[-1,:], state, v_first

def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x

def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x[-1,:]

class RWKV_x070(torch.nn.Module):
    def __init__(self, model):
        global DTYPE, DEVICE
        super().__init__()
        self.eval()
        self.model = model

        z = self.model
        self.n_head = z['blocks.0.att.r_k'].shape[0]
        self.head_size = z['blocks.0.att.r_k'].shape[1]
        self.vocab_size = z['emb.weight'].shape[0]
        self.n_embd = z['emb.weight'].shape[1]
        self.n_layer = 0
        keys = list(z.keys())
        for k in keys:
            layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
            self.n_layer = max(self.n_layer, layer_id+1)
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE)
            if k.endswith('att.r_k'):
                z[k] = z[k].flatten()

        z['emb.weight'] = F.layer_norm(z['emb.weight'],
                                       (self.n_embd,),
                                       weight=z['blocks.0.ln0.weight'],
                                       bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

    def forward_one(self, idx,
                    state0:List[torch.Tensor],
                    state1:List[torch.Tensor],
                    state2:List[torch.Tensor]):
        z = self.model
        x = z['emb.weight'][idx]

        v_first = torch.empty_like(x)
        for i in range(self.n_layer):
            bbb = f'blocks.{i}.'
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'

            xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

            xx, state0[i], state1[i], v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state0[i], v_first, state1[i],
                z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                z[att+'ln_x.weight'], z[att+'ln_x.bias'])
            x = x + xx

            xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

            xx, state2[i] = RWKV_x070_CMix_one(xx, state2[i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
            x = x + xx

        x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x, state0, state1, state2

    def forward_seq(self, idx,
                    state0:List[torch.Tensor],
                    state1:List[torch.Tensor],
                    state2:List[torch.Tensor]):
        z = self.model
        x = z['emb.weight'][idx]

        v_first = torch.empty_like(x)
        for i in range(self.n_layer):
            bbb = f'blocks.{i}.'
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'

            xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

            xx, state0[i], state1[i], v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state0[i], v_first, state1[i],
                z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                z[att+'ln_x.weight'], z[att+'ln_x.bias'])
            x = x + xx

            xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

            xx, state2[i] = RWKV_x070_CMix_seq(xx, state2[i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
            x = x + xx
        
        x = x[-1,:]
        x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x, state0, state1, state2

class wrapper_forward_one(torch.nn.Module):
    def __init__(self, origin_model):
        super().__init__()
        self.model = origin_model

    def forward(self, idx,
                state0:List[torch.Tensor],
                state1:List[torch.Tensor],
                state2:List[torch.Tensor]):
        return self.model.forward_one(idx, state0, state1, state2)

class wrapper_forward_seq(torch.nn.Module):
    def __init__(self, origin_model):
        super().__init__()
        self.model = origin_model

    def forward(self, idx,
                state0:List[torch.Tensor],
                state1:List[torch.Tensor],
                state2:List[torch.Tensor]):
        return self.model.forward_seq(idx, state0, state1, state2)

model = torch.load(args.model_path, map_location=DEVICE)
tokenizer = TRIE_TOKENIZER('../tokenizer/rwkv_vocab_v20230424.txt')
rwkv = RWKV_x070(model)
prompt_mode = "chat"

print(f"Layer num:{rwkv.n_layer}")
print(f"Hidden size:{rwkv.n_embd}")
print(f"Head size:{rwkv.head_size}")
print(f"Vocab size:{rwkv.vocab_size}\n")

def convert_forward_one():
    forward_one = wrapper_forward_one(rwkv)
    idx = torch.arange(1, dtype=torch.int32)
    state0 = []
    state1 = []
    state2 = []
    for i in range(rwkv.n_layer):
        state0.append(torch.zeros(rwkv.n_embd))
        state1.append(torch.zeros((rwkv.n_embd // rwkv.head_size,
                                rwkv.head_size, rwkv.head_size)))
        state2.append(torch.zeros(rwkv.n_embd))

    input_names = ['idx']
    output_names = ['out']
    for i in range(rwkv.n_layer):
        input_names.append(f'input_state0_{i}')
        output_names.append(f'output_state0_{i}')
    for i in range(rwkv.n_layer):
        input_names.append(f'input_state1_{i}')
        output_names.append(f'output_state1_{i}')
    for i in range(rwkv.n_layer):
        input_names.append(f'input_state2_{i}')
        output_names.append(f'output_state2_{i}')

    torch.onnx.export(
        forward_one, (idx, state0, state1, state2),
        f'{folder}/one/rwkv_forward_one.onnx',
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        opset_version=15)

def convert_forward_seq():
    forward_seq = wrapper_forward_seq(rwkv)
    idx = torch.arange(CHUNK_LEN, dtype=torch.int32)
    state0 = []
    state1 = []
    state2 = []
    for i in range(rwkv.n_layer): 
        state0.append(torch.zeros(rwkv.n_embd))
        state1.append(torch.zeros((rwkv.n_embd // rwkv.head_size,
                                rwkv.head_size, rwkv.head_size)))
        state2.append(torch.zeros(rwkv.n_embd))

    input_names = ['idx']
    output_names = ['out']
    for i in range(rwkv.n_layer):
        input_names.append(f'input_state0_{i}')
        output_names.append(f'output_state0_{i}')
    for i in range(rwkv.n_layer):
        input_names.append(f'input_state1_{i}')
        output_names.append(f'output_state1_{i}')
    for i in range(rwkv.n_layer):
        input_names.append(f'input_state2_{i}')
        output_names.append(f'output_state2_{i}')

    torch.onnx.export(
        forward_seq, (idx, state0, state1, state2),
        f'{folder}/seq/rwkv_forward_seq.onnx',
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        opset_version=15)

def test_net():
    ############ generation config ############
    GEN_TEMP = 1.0
    GEN_TOP_P = 0.3
    GEN_alpha_presence = 0.5
    GEN_alpha_frequency = 0.5
    GEN_penalty_decay = 0.996
    ###########################################

    # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
    state0_init = []
    state1_init = []
    state2_init = []
    for i in range(rwkv.n_layer): 
        state0_init.append(torch.zeros(rwkv.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE))
        state1_init.append(torch.zeros((rwkv.n_embd // rwkv.head_size, rwkv.head_size, rwkv.head_size), dtype=torch.float, requires_grad=False, device=DEVICE))
        state2_init.append(torch.zeros(rwkv.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE))

    model_tokens = []
    model_state = [state0_init, state1_init, state2_init]

    if STATE != None:
        # if loading a state, change sample strategy
        GEN_TOP_P = 0.2
        GEN_alpha_presence = 0.3
        GEN_alpha_frequency = 0.3

        state_raw = torch.load(STATE)
        for i in range(rwkv.n_layer):
            state1_init[i] = state_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=DEVICE).requires_grad_(False).contiguous()
        model_state = [state0_init, state1_init, state2_init]

    # use initial prompt if we are not loading a state
    else:
        init_ctx = "User: hi" + "\n\n"
        init_ctx += "Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it." + "\n\n"
        init_ctx.replace("\r\n", "\n")
        print(init_ctx, end="")

        tokens = tokenizer.encode(init_ctx)
        tokens = [int(x) for x in tokens]
        model_tokens += tokens
        while len(tokens) > 0:
            out, model_state[0], model_state[1], model_state[2] = rwkv.forward_seq(tokens[:CHUNK_LEN], model_state[0], model_state[1], model_state[2])
            tokens = tokens[CHUNK_LEN:]

    while True:
        if prompt_mode == "chat":
            msg = input("User: ")
            msg = re.sub(r"\n+", "\n", msg.strip())
            if msg == 'q' or msg == 'exit':
                break
            ctx = "User: " + msg + "\n\nAssistant:"
            print("\nAssistant:", end="")

        elif prompt_mode == "instruction":
            ins = input("Instruction: ")
            if ins == 'q' or ins == 'exit':
                break
            ins = re.sub(r"\n+", "\n", ins.strip())
            inp = input("\nInput: ")
            ins = re.sub(r"\n+", "\n", inp.strip())
            ctx = "Instruction: " + ins + "\n\nInput: " + inp + "\n\nResponse:"
            print("\nResponse:", end="")

        tokens = tokenizer.encode(ctx)
        model_tokens += tokens
        while len(tokens) > 0:
            prefill_tokens = tokens[:CHUNK_LEN]
            out, model_state[0], model_state[1], model_state[2] = rwkv.forward_seq(prefill_tokens, model_state[0], model_state[1], model_state[2])
            tokens = tokens[CHUNK_LEN:]

        occurrence = {}
        out_tokens = []
        out_last = 0

        for i in range(99999):
            for n in occurrence:
                # repetition penalty
                out[n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency 
            
            # disable END_OF_TEXT
            out[0] -= 1e10

            token = tokenizer.sample_logits(out, temperature=GEN_TEMP, top_p=GEN_TOP_P)

            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

            out, model_state[0], model_state[1], model_state[2] = rwkv.forward_one(token, model_state[0], model_state[1], model_state[2])
            model_tokens += [token]
            out_tokens += [token]

            tmp = tokenizer.decode(out_tokens[out_last:])            
            # only print & update out_last when it's a valid utf-8 string and not ending with \n
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                print(tmp, end="", flush=True)
                out_last = i + 1

            if "\n\n" in tmp:
                print(tmp, end="", flush=True)
                break

if args.test:
    test_net()
else:
    convert_forward_one()
    convert_forward_seq()