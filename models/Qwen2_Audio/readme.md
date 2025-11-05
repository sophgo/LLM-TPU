Qwen2-Audio（算能 AirBox 边缘端）实施说明

概述
- 本项目在算能（SOPHON）AirBox 边缘端部署并运行 Qwen2-Audio 模型，提供 Python 与 C++ 两条推理路径。
- 工程包含语音特征提取、文本与音频融合、KV-Cache 预填充（prefill），以及逐 Token 生成（greedy）的完整链路。
- 依赖算能 BMRuntime/LibSophon 库，支持 PCIe/SOC 两种目标平台；OpenCV 为可选依赖。

目录结构
- `compile/`：编译脚本等。
- `python_demo/`：Python 入口与 PyBind C++ 模块源码
  - `pipeline.py`：Python 推理管线（文本/音频处理、prefill、生成循环）
  - `chat.cpp`：PyBind 导出的 C++ 推理实现（BMRuntime 图执行、KV 缓存）
  - `utils.py`：音频处理、掩码与合并工具函数
  - `CMakeLists.txt`：构建 `chat` Python 模块

环境准备
- 算能软件栈：确保安装 `libsophon-current` 与 BMRuntime，常见头文件与库位于：`/opt/sophon/libsophon-current`
- 构建工具：`cmake >= 3.10`，`g++` 支持 C++17
- 可选依赖：OpenCV（如果未安装，C++ CMake 已做“可选”处理）
- Python 运行环境：`Python 3.8+`，建议安装以下依赖：
  - `torch`、`numpy`、`librosa`
  - `modelscope`（用于本地下载/加载模型配置与处理器）

模型与资源
- `pipeline.py` 使用 `modelscope` 的 `snapshot_download` 并设置 `local_files_only=True`，需提前在本地缓存好模型资源（Qwen/Qwen2-Audio-7B-Instruct）。
- 推理所需 bmodel（权重/图）请准备在本地，并在运行时通过参数传入。

构建与运行（Python Demo）
- 构建 PyBind 模块：
  - `cd python_demo`
  - `mkdir build && cd build`
  - `cmake ..`
  - `make -j$(nproc)`
- 运行 Python 管线：
  - 将生成的 `chat.*.so` 模块所在目录加入 `PYTHONPATH`
    - `export PYTHONPATH=$(pwd):$PYTHONPATH`（在 `python_demo/build` 目录下）
  - 运行：
    - `cd ..`
    - `python3 pipeline.py -m <bmodel_path> -d <devid> -c ../config`
  - 交互：程序会提示输入文本或音频路径（本地文件）。
- 关键流程（与代码对应）：
  - 文本嵌入：`forward_embed(input_ids)` → 形状约为 `(1, 599, 4096)`
  - 音频注意力掩码构建：依据输入波形长度与特征提取长度计算，填充为 `-inf`/`0` 的 4D 掩码
  - 音频前向：`forward_audio(input_features, audio_attention_mask)` → `forward_project(...)` 得到 `(1, 750, 4096)`
  - 合并：将音频嵌入写入 `<AUDIO>` 占位的序列位置，并同步更新 `attention_mask` 与 `position_ids`
  - Prefill：将三者扩展/对齐到 599 的最大长度，按层运行 Block 前向，收集 `k_caches`/`v_caches` 与 `inputs_embeds`
  - 生成：逐 Token 使用 `forward_embed_cache(...)` 与缓存，配合贪心头进行生成

设备与库说明
- 目标平台：根据 CMake 的 `SOC_TARGET`/`PCIE_TARGET` 条件，选择链接的库与包含目录
- 关键库：`bmlib`、`bmrt`、`pthread`、`dl`，以及项目内第三方 `tokenizers_cpp`/`tokenizers_c`
- 头文件与库路径：默认使用 `/opt/sophon/libsophon-current/include` 与 `/opt/sophon/libsophon-current/lib`

特殊 Token 映射
- 文本结束：`<|end|>` → `ID_END`
- 音频结束：`<|endoftext|>` → `ID_AU_END`
- 请确保 Tokenizer 与模型一致，以避免生成过程提前/滞后终止。

常见问题与排查
- OpenCV 未找到：
  - 设置 `OpenCV_DIR` 指向 `OpenCVConfig.cmake` 所在目录，或将 OpenCV 安装前缀加入 `CMAKE_PREFIX_PATH`
  - 当前 C++ 工程允许无 OpenCV 构建，但部分图像相关示例将不可用
- Python 无法导入 `chat`：
  - 确认已构建并将生成的 `.so` 模块所在目录加入 `PYTHONPATH`
- 设备库缺失或运行失败：
  - 检查 `/opt/sophon/libsophon-current` 是否存在并包含 `bmruntime_interface.h`、`bmlib_runtime.h` 等
  - 确保设备 ID 正确（`-d <devid>`）且设备处于就绪状态
- 生成行为异常（如 `inputs_embeds` 与 KV 缓存不一致）：
  - 已在 C++ 侧修正 Block 前向路径，确保 `forward(...)` 使用逐层 `net_blocks[idx]` 与缓冲清理
  - 请确认合并后的 `inputs_embeds/attention_mask/position_ids` 长度对齐为 599，且 dtype 分别为 `float32/int32`

性能与部署建议
- 使用 `-j$(nproc)` 加速构建
- 提前缓存模型与处理器配置文件，减少运行时 IO
- 根据设备内存与算力，合理选择批大小与最大序列长度

示例命令
- 构建并运行 Python：
  - `cd /data/LLM-TPU/models/Qwen2_Audio/python_demo`
  - `mkdir build && cd build && cmake .. && make -j$(nproc)`
  - `export PYTHONPATH=$(pwd):$PYTHONPATH`
  - `cd .. && python3 pipeline.py -m /path/to/qwen2_audio.bmodel -d 0 -c ../config`

版权与许可证
- 本说明文件不包含许可证内容。请根据上游模型与依赖库的授权条款合规使用与分发。
