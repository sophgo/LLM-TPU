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


if __name__ == "__main__":
    providers = ["CPUExecutionProvider"]
    model_path = "tmp/onnx/"

    block_inputs_0 = np.load("tmp/bad_case_npz/g_bmodel_decode_input_0.npz")
    block_inputs_0 = dict(block_inputs_0)
    block_inputs_0["position_ids"] = block_inputs_0["position_ids"].astype(np.int64)

    block_out = test_block_cache_i(block_inputs_0, 0)

    print(block_out)
