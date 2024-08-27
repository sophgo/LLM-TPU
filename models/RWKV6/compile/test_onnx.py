import onnx

import onnxruntime as ort
import numpy as np


def softmax(x, axis=None):
    # 沿指定轴计算指数值
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    # 沿指定轴计算归一化指数值
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    # 计算softmax值
    softmax_x = exp_x / sum_exp_x
    return softmax_x


def sample_logits(
    out: np.ndarray, temperature: float = 1.0, top_p: float = 0.8
) -> list[list[int]]:
    """
    对模型输出的logits进行采样。
    Args:
        out (np.ndarray): 模型输出的logits张量，形状为[Batch, vocab_size]。
        temperature (float): 温度参数，用于调节采样的多样性，默认为1.0。
        top_p (float): Top-p截断参数，用于稳定和控制采样概率分布，默认为0.8。

    Returns:
        list[list[int]]: 采样结果，每个子列表包含一个样本中的词的索引序号。
    """
    # 将out转换为概率分布
    probs = softmax(out, axis=-1)
    # 对每个样本进行采样
    sampled_indices = []
    for sample_probs in probs:
        # 根据top_p截断概率分布
        sorted_probs = np.sort(sample_probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        sample_probs[sample_probs < cutoff] = 0
        # 对概率分布进行温度调节
        if temperature != 1.0:
            sample_probs = np.power(sample_probs, 1.0 / temperature)
        # 归一化概率分布
        sample_probs /= np.sum(sample_probs)
        # 从概率分布中采样一个索引
        sampled_index = np.random.choice(a=len(sample_probs), p=sample_probs)
        sampled_indices.append([sampled_index])
    # 返回采样结果
    return sampled_indices


def model_print(model_file):
    model = onnx.load(model_file)
    # 打印模型的输入参数
    for input in model.graph.input:
        print("Input name:", input.name)
        print("Input shape:", [d.dim_value for d in input.type.tensor_type.shape.dim])
        print()

    for output in model.graph.output:

        print("Output name:", output.name)
        print("Output shape:", [d.dim_value for d in output.type.tensor_type.shape.dim])
        print()


def test_emb(input_dict, path="tmp/onnx/"):
    model_p = f"{path}embedding.onnx"
    session = ort.InferenceSession(model_p, providers=providers)
    ort_outs = session.run(None, input_dict)
    print(f"emb {ort_outs[0].tolist()[0][0:100]}\r\n")
    return ort_outs[0]


def test_block_i(input_dict, id, path="tmp/onnx/"):
    model_p = f"{path}block_{id}.onnx"
    session = ort.InferenceSession(model_p, providers=providers)
    ort_outs = session.run(None, input_dict)
    # print(f"block {id} {ort_outs}")
    print(f"block {id}")
    return ort_outs


def test_lm_head(input_dict, path="tmp/onnx/"):
    model_p = f"{path}lm_head.onnx"
    session = ort.InferenceSession(model_p, providers=providers)
    ort_outs = session.run(None, input_dict)
    print(f"lm_head {ort_outs}")
    return ort_outs[0]


if __name__ == "__main__":
    providers = ["CPUExecutionProvider"]
    model_path = "tmp/onnx/"

    # EMB
    token = np.array([[34550]]).astype(np.int64).transpose()
    t = token[0]
    emb_inputs = {"input_ids": t.reshape(1, 1)}
    emb_out = test_emb(emb_inputs)
    # print(emb_out.tolist())
    state = np.zeros((1, 1584, 2048), dtype=np.float32)

    # Block
    model_print("tmp/onnx/block_0.onnx")
    for i in range(24):
        if i == 0:
            block_inputs_0 = {"b_in": emb_out, "state.1": state}
            # print(state[:,500])
            # print(state[:,500])
            block_out = test_block_i(block_inputs_0, i)
            print(f"state {block_out[0]}")
            print(f"state {np.min(block_out[1])} {np.max(block_out[1])}")
            # print(f"state {block_out[1][:,500]}")
            # print(f"state {block_out[1]}")
        else:
            block_inputs_i = {"b_in": block_out[0], "state.1": block_out[1]}
            block_out = test_block_i(block_inputs_i, i)
            print(f"state {np.min(block_out[1])} {np.max(block_out[1])}")
            # print(f"state {block_out[1][:,500]}")
            print(f"state {np.min(block_out[0])} {np.max(block_out[0])}")
            print(f"state {block_out[0]}")
            # print(f"state {block_out[1]}")

    # Lm_head
    # model_print("tmp/onnx/lm_head.onnx")
    # print(f"lm_head_in {block_out[0]}")
    lm_head_inputs = {"hidden_dim": block_out[0]}
    lm_head_out = test_lm_head(lm_head_inputs)
    print(f"logits {lm_head_out.tolist()[0][0:60]}")
    print(f"state {np.min(lm_head_out)} {np.max(lm_head_out)}")
    
    token_id = sample_logits(lm_head_out, 1, 0)
    print(f"token_id {token_id}")