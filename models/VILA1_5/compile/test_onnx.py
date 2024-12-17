import onnx

import onnxruntime as ort
import numpy as np

def test_block_i(input_dict, id, path="tmp/onnx/"):
    model_p = f"{path}block_{id}.onnx"
    session = ort.InferenceSession(model_p, providers=providers)
    ort_outs = session.run(None, input_dict)
    # print(f"block {id} {ort_outs}")
    print(f"block {id}")
    return ort_outs

def test_block_cache_i(input_dict, id, path="tmp/onnx/"):
    model_p = f"{path}block_cache_{id}_seewo.onnx"
    session = ort.InferenceSession(model_p, providers=providers)
    ort_outs = session.run(None, input_dict)
    # print(f"block {id} {ort_outs}")
    print(f"block cache {id}")
    return ort_outs


def test_lm_head(input_dict, path="tmp/onnx/"):
    model_p = f"{path}lm_head.onnx"
    session = ort.InferenceSession(model_p, providers=providers)
    ort_outs = session.run(None, input_dict)
    print(f"lm_head {ort_outs}")
    return ort_outs[0]

def cosine_similarity(matrix1, matrix2):
    matrix1 = matrix1.flatten()
    matrix2 = matrix2.flatten()
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape."

    dot_product = np.dot(matrix1.flatten(), matrix2.flatten())
    norm_matrix1 = np.linalg.norm(matrix1)
    norm_matrix2 = np.linalg.norm(matrix2)
    similarity = dot_product / (norm_matrix1 * norm_matrix2)

    return similarity


if __name__ == "__main__":
    providers = ["CPUExecutionProvider"]
    model_path = "tmp/onnx/"

    block_inputs = np.load(f"tmp/test_block/bmodel_input_1.npz")
    block_inputs = dict(block_inputs)
    block_inputs["position_ids"] = block_inputs["position_ids"].astype(np.int64)

    for i in range(1,32):
        block_out = test_block_i(block_inputs, i)
        torch_out = np.load(f"tmp/test_block/torch_block_output_{i}.npz")

        seq_len = torch_out["hidden_states"].shape[1]
    
        bmodel_output_0 = block_out[0][:,seq_len-1]
        torch_output_0 =torch_out["hidden_states"][:,seq_len-1]

        cos_sim_0 = cosine_similarity(bmodel_output_0, torch_output_0)
        print(f"Layer {i} : {cos_sim_0}")
        block_inputs["input_states"][:,:seq_len] = block_out[0][:,:seq_len]
