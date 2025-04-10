
0. 创建多个stage的文件夹，官网拉取模型
mkdir stage_1 && mkdir stage_2 && mkdir stage_3

1. 依次进入stage_i ，导onnx模型
python model_export.py --quantize w4bf16 --tpu_mlir_path /workspace/tpu-mlir/ --model_path /workspace/models/Qwen2-VL-2B-Instruct --seq_length 1280(目标seq_length) --input_seq_len 800(目标input_seq_len) --visual_length 600 --out_dir qwen2_vl_2b --embedding_disk --not_compile(不做编译)

2. 编译bmodel
cd tpu-mlir && source envsetup.sh
source compile_multi.sh --name qwen2-vl-2b --seq_length (给定seq_len)

3. combine 包含 block 、 block_cache 、 penalty_sample_head 三个模块的 bmodel
model_tool --combine v1.bmodel v2.bmodel v3.bmodel -o p.bmodel # 合并block 、 block_cache 、 penalty_sample_head 三个模块， v1/v2/v3 按照 seq_len 从小到大排列

4. combine 包含 lm_head 、 vit 、 greedy_head 的 bmodel
model_tool --combine p.bmodel lm_head.bmodel vit.bmodel greedy_head.bmodel -o final.bmodel # 任选一个stage即可，只合并一次
