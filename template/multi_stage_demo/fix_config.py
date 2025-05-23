import json
import ast

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
            with open(f"config_data{layer_ids[-2]}.json", "w", encoding="utf-8") as file:
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
                    info["prun_dim"] = i
                    break
        elif data["layer_name"].endswith("up_proj.weight"):
            info["idx"] = 5
            for i in range(2):
                if in_shape[i] != prun_shape[i]:
                    info["prun_dim"] = i
                    break
        elif data["layer_name"].endswith("down_proj.weight"):
            info["idx"] = 6
            for i in range(2):
                if in_shape[i] != prun_shape[i]:
                    info["prun_dim"] = i
                    break
        list_c = ast.literal_eval(data["pruned_channel"])
        info["pruned_channel"] = sorted(list_c)
        save_dict["MatMul"].append(info)
