import onnx
import numpy as np
import json

model = onnx.load("block_0.onnx")
with open('data.json', 'r', encoding='utf-8') as f:
    config_data = json.load(f)

config_data["MatMul"][0]["t_name"] = "mlp.gate_proj.weight"
config_data["MatMul"][1]["t_name"] = "mlp.up_proj.weight"
config_data["MatMul"][2]["t_name"] = "mlp.down_proj.weight"

for tensor in model.graph.initializer:
    if "mlp.gate_proj.weight" == tensor.name:
        data = onnx.numpy_helper.to_array(tensor)
        data = data.copy()  # 关键：复制为可写副本
        for i in range(len(data)):
            for j in range(len(data[0])):
                if j in config_data["MatMul"][0]["pruned_channel"]:
                    data[i][j] = 0
        new_tensor = onnx.numpy_helper.from_array(data, tensor.name)
        tensor.CopyFrom(new_tensor)
    elif "mlp.up_proj.weight" == tensor.name:
        data = onnx.numpy_helper.to_array(tensor)
        data = data.copy()  # 关键：复制为可写副本
        for i in range(len(data)):
            for j in range(len(data[0])):
                if j in config_data["MatMul"][1]["pruned_channel"]:
                    data[i][j] = 0
        new_tensor = onnx.numpy_helper.from_array(data, tensor.name)
        tensor.CopyFrom(new_tensor)
    elif "mlp.down_proj.weight" == tensor.name:
        data = onnx.numpy_helper.to_array(tensor)
        data = data.copy()  # 关键：复制为可写副本
        for i in range(len(data)):
            if i in config_data["MatMul"][2]["pruned_channel"]:
                data[i, :] = 0
        new_tensor = onnx.numpy_helper.from_array(data, tensor.name)
        tensor.CopyFrom(new_tensor)

onnx.save(model, "pruned_model.onnx")
