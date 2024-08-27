# Reference from https://github.com/yuunnn-w/RWKV_Pytorch

import argparse
import os

import torch
import torch.nn.functional as F

torch.set_printoptions(profile="default", edgeitems=50, linewidth=200)
from tqdm import tqdm

from src.model import RWKV_RNN


def sample_logits(
    out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8
) -> torch.Tensor:
    """
    对模型输出的logits进行采样。

    Args:
        out (torch.Tensor): 模型输出的logits张量,形状为[Batch, vocab_size]。
        temperature (float): 温度参数,用于调节采样的多样性,默认为1.0。
        top_p (float): Top-p截断参数,用于稳定和控制采样概率分布,默认为0.8。

    Returns:
        torch.Tensor: 采样结果,形状为[Batch, 1],每个元素表示一个样本中采样得到的词的索引。
    """
    # 确保top_p和temperature都是非负值
    top_p = max(0.0, min(1.0, top_p))
    temperature = max(0.0, temperature)

    # 将out转换为概率分布
    probs = F.softmax(out, dim=-1)

    # 根据top_p截断概率分布
    sorted_probs, _ = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_mask = (cumulative_probs > top_p).float()
    cutoff_index = torch.argmax(
        cutoff_mask
        * torch.arange(cutoff_mask.shape[-1], device=cutoff_mask.device).float(),
        dim=-1,
    )
    cutoff_values = sorted_probs.gather(-1, cutoff_index.unsqueeze(-1)).squeeze(-1)
    probs = torch.where(
        probs < cutoff_values.unsqueeze(-1), torch.zeros_like(probs), probs
    )

    # 对概率分布进行温度调节
    if temperature != 1.0:
        probs = torch.pow(probs, 1.0 / temperature)

    # 归一化概率分布
    probs /= torch.sum(probs, dim=-1, keepdim=True)

    # 如果top_p为0,则选择概率最大的位置;否则按照概率分布随机采样
    if top_p != 0:
        sampled_indices = torch.multinomial(probs, num_samples=1)
    else:
        sampled_indices = torch.argmax(probs, dim=-1, keepdim=True)

    return sampled_indices


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        emb_out = origin_model.emb(input_ids).squeeze(1)
        ln_out = origin_model.manual_layer_norm(
            emb_out, origin_model.ln0_weight, origin_model.ln0_bias, 1e-5
        )
        return ln_out


def convert_embedding():
    model = Embedding()
    input_ids = torch.zeros(model_args["batch_size"], 1).long()

    torch.onnx.export(
        model,
        (input_ids),
        f"{folder}/embedding.onnx",
        verbose=False,
        input_names=["input_ids"],
        output_names=["hidden_dim"],
        do_constant_folding=True,
        opset_version=15,
    )


def test_emb():
    model = Embedding()
    input_ids = torch.tensor([[74]]).long()
    out = model(input_ids)
    print(f"out {out} {out.shape}")


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = origin_model.blocks[layer_id]

    def forward(self, b_in, state, b_id):
        b_out, state = self.layer(b_in, state, b_id)
        return b_out, state


def convert_block(layer_id, verbose=False):
    model = Block(layer_id)
    b_in = torch.zeros(model_args["batch_size"], EMB_DIM)
    state = torch.randn(model_args["batch_size"], *STATE_SIZE)
    b_id = torch.tensor(layer_id).long()
    # b_id = layer_id

    torch.onnx.export(
        model,
        (b_in, state, b_id),
        f"{folder}/block_{layer_id}.onnx",
        verbose=verbose,
        input_names=["b_in", "state", "b_id"],
        output_names=["b_out", "state"],
        do_constant_folding=True,
        opset_version=15,
    )


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, b_out):
        head_in = origin_model.manual_layer_norm(
            b_out, origin_model.ln_out_weight, origin_model.ln_out_bias, 1e-5
        )
        m_logits = origin_model.head(head_in)
        return m_logits


def test_lm_head():
    model = LmHead()
    input = torch.randn(model_args["batch_size"], 1, EMB_DIM)
    out = model(input)
    print(out)


def convert_lm_head():
    model = LmHead()
    input = torch.randn(model_args["batch_size"], EMB_DIM)

    torch.onnx.export(
        model,
        (input),
        f"{folder}/lm_head.onnx",
        verbose=False,
        input_names=["hidden_dim"],
        output_names=["m_logits"],
        do_constant_folding=True,
        opset_version=15,
    )


# refs:https://github.com/sophgo/LLM-TPU/blob/main/models/Llama2/compile/export_onnx.py
class GreedyHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, m_logits):
        _, token = torch.topk(m_logits.float(), 1)
        return token


def convert_greedy_head():
    model = GreedyHead()
    m_logits = torch.randn(1, VOCAB_SIZE)

    torch.onnx.export(
        model,
        (m_logits),
        f"{folder}/greedy_head.onnx",
        verbose=False,
        input_names=["m_logits"],
        output_names=["token"],
        do_constant_folding=True,
        opset_version=15,
    )


# refs:https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
class PenaltySampleHead(torch.nn.Module):

    def __init__(self, top_k=50, min_tokens_to_keep=5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, : self.min_tokens_to_keep] = True

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
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.0]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token


def convert_penalty_sample_head():
    model = PenaltySampleHead()
    m_logits = torch.randn(1, VOCAB_SIZE)
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    top_p = torch.tensor([0.8])
    temperature = torch.tensor([0.98])
    penalty = torch.tensor([0.98])

    torch.onnx.export(
        model,
        (m_logits, input_ids, top_p, temperature, penalty),
        f"{folder}/penalty_sample_head.onnx",
        verbose=False,
        input_names=["m_logits", "input_ids", "top_p", "temperature", "penalty"],
        output_names=["probs", "token"],
        do_constant_folding=True,
        opset_version=15,
    )


def test_all(token_id, state):
    embedding = Embedding()
    x = embedding(token_id)
    print(f"emb out =\n{x}")

    for i in range(NUM_LAYERS):
        block = Block(i)
        i = torch.tensor([i]).long()
        x, state = block(x, state, i)
        # print(f"\nblock {i} out =\n{x}\n")
        print(f"block {i} state =\n{torch.min(state)}\n{torch.max(state)}")
        # print(f"block {i} state =\n{state}")

    lm_head = LmHead()
    logits = lm_head(x)
    print(logits.shape)
    print(f"logits {torch.min(logits)}\n{torch.max(logits)}")
    new_token_id = sample_logits(logits[0], 1, 0)
    print(new_token_id)
    print(logits[0][new_token_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export onnx.")
    parser.add_argument("--model_path", "-m", type=str, help="path to the torch model.")
    parser.add_argument("--test", "-t", action="store_true", help="enable some tests.")
    # parser.add_argument('--model_path', type=str, help='path to the torch model.')
    parser.add_argument("--seq_length", type=int, default=4096, help="sequence length")

    args = parser.parse_args()

    model_path = args.model_path
    folder = f"./tmp/onnx"
    # create folder to store onnx
    if not os.path.exists(folder):
        os.makedirs(folder)

    model_args = {
        "MODEL_NAME": model_path,  # 模型文件的名字，pth结尾的权重文件。
        "vocab_size": 65536,  # 词表大小
        "batch_size": 1,
    }
    print(f"Loading model {model_args['MODEL_NAME']}.pth...")
    origin_model = RWKV_RNN(model_args)

    VOCAB_SIZE = origin_model.args["vocab_size"]
    NUM_LAYERS = origin_model.num_layer
    EMB_DIM = origin_model.n_embd
    SEQ_LENGTH = args.seq_length
    print("EMB_DIM")
    print(EMB_DIM)
    STATE_SIZE = origin_model.state_size
    print(origin_model)
    print("Done.")

    origin_model.eval()  # 确保模型处于评估模式
    for param in origin_model.parameters():
        param.requires_grad = False

    # 准备输入数据的示例
    # example_token = torch.zeros(
    #     model_args["batch_size"], 1
    # ).long()  # token输入的尺寸 [batch, 1]
    example_token = torch.tensor([[1922]]).long()  # token "hi"
    example_state = torch.zeros(
        model_args["batch_size"], *origin_model.state_size
    )  # state_size是state输入的尺寸
    # 测试推理
    A, B = origin_model(example_token, example_state)

    print(f"output is {sample_logits(A,1,0)}")
    # print(f"state is {B}")
    if args.test:
        test_all(
            torch.tensor([[1922]]).long(),
            torch.zeros(model_args["batch_size"], *origin_model.state_size),
        )
        # test_emb()
        # test_lm_head()
        exit(0)

    # 导出模型
    print("\nExport Onnx...")

    print(f"Convert block & block_cache")
    for i in tqdm(range(NUM_LAYERS)):
        convert_block(i)
        # exit()

    print(f"Convert embedding")
    convert_embedding()

    print(f"Convert lm_head")
    convert_lm_head()
    convert_greedy_head()
    convert_penalty_sample_head()

    print(f"\nDone.\nOnnx weight has saved in {folder}")
