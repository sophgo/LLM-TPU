import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import random
import subprocess
import json
import ast

# gen config files
with open('qwen2-vl-2b.json', 'r', encoding='utf-8') as file:
    data_list = json.load(file)
save_dict = {"MatMul": []}
layer_ids = []
for data in data_list:
    if not data["layer_name"].startswith("model.layers"):
        continue
    layer_id = int(data["layer_name"].split(".")[2])
    if len(layer_ids) == 0 or (len(layer_ids) != 0 and layer_id != layer_ids[-1]):
        layer_ids.append(layer_id)
        if len(layer_ids) != 1:
            with open(f"pruning_config_{layer_ids[-2]}.json", "w", encoding="utf-8") as file:
                json.dump(save_dict, file)
                save_dict = {"MatMul": []}
    if data["layer_name"].startswith(f"model.layers.{layer_ids[-1]}.mlp"):
        info = {"idx": 0, "prun_dim": 0, "pruned_channel": [0] }
        in_shape = ast.literal_eval(data["original_shape"])
        prun_shape = ast.literal_eval(data["pruned_shape"])
        if data["layer_name"].endswith("gate_proj.weight"):
            info["idx"] = 4
            for i in range(2):
                if in_shape[i] != prun_shape[i]:
                    info["prun_dim"] = 1 - i
                    break
        elif data["layer_name"].endswith("up_proj.weight"):
            info["idx"] = 5
            for i in range(2):
                if in_shape[i] != prun_shape[i]:
                    info["prun_dim"] = 1 - i
                    break
        elif data["layer_name"].endswith("down_proj.weight"):
            info["idx"] = 6
            for i in range(2):
                if in_shape[i] != prun_shape[i]:
                    info["prun_dim"] = 1 - i
                    break
        list_c = ast.literal_eval(data["pruned_channel"])
        info["pruned_channel"] = sorted(list_c)
        save_dict["MatMul"].append(info)

# gen block_0 test_input
input_states = torch.rand((1, 960, 1536), dtype=torch.float32)
position_ids = torch.randn(1, 960).clamp(0, 100)
attention_mask = torch.rand((1, 1, 960, 960), dtype=torch.float32)
np.savez("block_model_in_f32.npz", input_states=input_states, position_ids=position_ids, attention_mask=attention_mask)

subprocess.run('model_runner.py --input block_model_in_f32.npz --model pruned_model.onnx --output block_model_ref_outputs.npz', shell=True)
data = np.load("block_model_ref_outputs.npz")
old_keys = ['hidden_states', 'past_k', 'past_v']
new_keys = ['hidden_states_Add', 'past_k_Add', 'past_v_Reshape']
new_data = {new: data[old] for old, new in zip(old_keys, new_keys)}
np.savez('block_model_ref_outputs.npz', **new_data)