# Step 03: Pipeline Implementation

> **文件落点**：per-model 文件在 `<work_dir>` = `/workspace/llm/LLM-TPU/models/<model>/tmp/` 下（见 SKILL.md）。
>
> **前置条件**：步骤 2 完成，bmodel 编译成功。
> **产物**：`chat.cpp` + `pipeline.py` + 端到端跑通推理

## 目标

在 LLM-TPU 仓库实现推理 pipeline，让 bmodel 能实际运行。
包含 C++ pybind11 封装和 Python 推理脚本。

按 `<model>_plan.md` 中的适配顺序执行：**文本 → 视觉（如有）→ 语音（如有）**。
如果某个模态在 Converter 中继承了已有架构，pipeline 代码从 LLM-TPU 仓库对应模型复制。

## 执行步骤

### 1. 创建目录结构

模型目录 `models/<Model>/` 已由步骤 0 创建（含 `tmp/` 子目录），本步骤在其下创建 `python_demo/` 等，**勿删 `tmp/`**。

```
models/<Model>/
├── README.md           # 步骤 5 写
├── config/             # 步骤 2 编译 bmodel 时生成，从输出目录移动过来
├── <model>.bmodel      # 步骤 2 编译生成，从输出目录移动过来
├── tmp/               # 步骤 0 已创建，per-model template + plan（已 .gitignore）
├── python_demo/
│   ├── CMakeLists.txt
│   ├── chat.cpp
│   ├── pipeline.py
│   └── test.jpg        # 测试图片（多模态模型需要，从其他 demo 拷贝）
```

> 注：`utils.h` 和 `cnpy.cpp` 不是标准目录的一部分。如果精度调试需要 dump 工具，
> 在步骤 4 从 `LLM-TPU/support/debug/` 目录拷贝，步骤 5 清理。

### 2. CMakeLists.txt

标准模板，只需改项目名：

```cmake
cmake_minimum_required(VERSION 3.10)
project(<ModelName>)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

add_definitions(--std=c++17 -fPIC -Wall -Werror)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)
set(pybind11_DIR "$ENV{HOME}/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(chat chat.cpp)
target_link_libraries(chat PUBLIC bmrt bmlib)
install(TARGETS chat DESTINATION python)
```

如需链接 `cnpy.cpp`（numpy 文件 I/O），在 `pybind11_add_module` 中加上 `cnpy.cpp`，
`target_link_libraries` 加上 `z`（zlib）。大多数模型不需要。

### 3. chat.cpp

#### 类结构

从最相似的已有模型复制 `chat.cpp`，然后修改：

1. **类名**：`Qwen3_5` → `<ModelName>`
2. **网络发现**：根据 bmodel 中实际包含的网络名调整
3. **forward_vit 签名**：按模型的 ViT 输入需求定义（见下表）
4. **文本模型部分**：如果文本可继承，`forward_first`/`forward_next`/`forward_embed` 通常无需改动

#### pybind11 接口契约 — 所有模型必须导出

```cpp
.def(pybind11::init<>())
.def("init",            &ModelName::init)
.def("deinit",          &ModelName::deinit)
.def("forward_embed",   &ModelName::forward_embed)
.def("forward_first",   &ModelName::forward_first)
.def("forward_next",    &ModelName::forward_next)
.def("clear_history",   &ModelName::clear_history)
.def_readonly("SEQLEN",           &ModelName::SEQLEN)
.def_readonly("MAX_INPUT_LENGTH", &ModelName::MAX_INPUT_LENGTH)
.def_readonly("support_history",  &ModelName::support_history)
.def_readonly("history_length",   &ModelName::history_length)
```

VLM 额外导出：
```cpp
.def("forward_vit",    &ModelName::forward_vit)
.def_readonly("MAX_PIXELS",   &ModelName::MAX_PIXELS)
.def_readonly("MAX_PATCHES",  &ModelName::MAX_PATCHES)
```

#### forward_vit 签名参考

根据模型的 ViT 输入复杂度选择：

```cpp
// 简单（无动态索引）：
void forward_vit(ArrayFloat const &pixel_values, int vit_offset);

// 中等（有位置编码）：
void forward_vit(ArrayFloat const &pixel_values, ArrayInt const &position_ids,
                 ArrayInt const &grid_thw, int vit_offset);

// 复杂（有动态空间索引）：
void forward_vit(ArrayFloat const &pixel_values, ArrayInt const &pos_ids,
                 ArrayInt const &reorder_index, ..., int vit_offset);
```

#### init() 网络发现

```cpp
// 标准流程：
auto net_names = bmrt_get_network_names(p_bmrt, &num_nets);
net_embed   = bmrt_get_network_info(p_bmrt, "embedding");
net_lm_head = bmrt_get_network_info(p_bmrt, "lm_head");

// ViT 网络：按名字查找
int num_vit = 0;
if (is_exist("vit", net_names, num_nets)) {
    net_vit = bmrt_get_network_info(p_bmrt, "vit");
    num_vit++;
}

// block 数量 = 总网络数 - embed - lm_head - embedding_cache - vit(s)
num_blocks = num_nets - 3 - num_vit;
```

#### 动态 vs 静态 shape 处理

根据模板中的编译模式选择，chat.cpp 需要区分静态和动态写法：

**检测方式**：
```cpp
is_dynamic = net_blocks[0]->is_dynamic;  // 从 bmodel 网络信息读取
```

**关键区别**：

| 区别点 | 静态 | 动态 |
|--------|------|------|
| attention mask | 固定分配 `MAX_INPUT_LENGTH²` | 按实际 `token_length²` 分配 |
| forward_first | 直接 `net_launch()`（shape 固定） | 先设实际 shape 再 `net_launch()` |
| ViT mask（VLM） | padding 到 `MAX_PATCHES²` | 按实际 patches 数设 shape |
| dev_buffer | embedding output 大小 | `SEQLEN * HIDDEN_SIZE`（支持历史时需要） |

**动态 shape 的写法**：
```cpp
// forward_first 中，每次 forward 前设置实际 shape：
in_tensors[0].shape.dims[1] = token_length;
in_tensors[1].shape.dims[1] = token_length;
if (prefill_mask) {
    in_tensors[2].shape.dims[2] = token_length;
    in_tensors[2].shape.dims[3] = token_length;
}
net_launch(net_blocks[idx], ...);
```

**静态 shape 的写法**：
```cpp
// shape 固定，不需要每次设置：
net_launch(net_blocks[idx], ...);
```

有些模型同时支持两种模式（如 Qwen3、Qwen2_5_VL），用 `if (is_dynamic)` 分支处理。
有些模型只有动态模式（如 Qwen3_5、MiniCPMV4_6），不需要分支。
**参考 `<model>_plan.md` 中的编译模式选择来确定写法。**

### 4. pipeline.py

> ⚠️ **pipeline 不得依赖源模型仓库（尤其权重 safetensors）**。原则上，**只传入 bmodel + config 就应该能跑通**——运行时不要去加载 HF 原始权重（多 GB safetensors），也不要 `from_pretrained` 整个模型目录只为拿权重。可预计算的小数据（位置编码、特殊索引表、常量矩阵等）在 Converter 里算好烘焙进 bmodel，或导出一个小 npz/json 放到 `config/`，pipeline 从 config 读。processor/tokenizer 从 config 目录加载不算权重依赖。

#### 标准类结构

```python
class <ModelName>():
    def __init__(self, args):
        self.model = chat.<ModelName>()
        self.model.init(self.device, args.model_path)
        self.processor = AutoProcessor.from_pretrained(args.config_path, ...)
        self.tokenizer = self.processor.tokenizer

        # 特殊 token ID — 必须从 tokenizer 读取，不要硬编码
        self.ID_xxx = self.tokenizer.convert_tokens_to_ids('<xxx>')

    def get_media_type(self, file_path): ...
    def process(self, input_str, media_type, media_path=""): ...
    def get_rope_index(self, input_ids, ...): ...
    def forward_prefill(self, position_ids): ...
    def run_once(self, input_str, media_path=""): ...
    def chat(self): ...
```

#### process() — 统一消息构建入口

> 💡 消息格式和 processor 调用参数请参考 HF 模型仓库的 README，里面通常有示例代码。

```python
def process(self, input_str, media_type, media_path=""):
    """构建消息并调用 processor 分词。"""
    # 1. 根据 media_type 构建 content（参考 HF README 的 message 格式）
    # 2. 调用 processor.apply_chat_template（参考 HF README 的参数）
    # 注意：transformers 5.x+ 需要将模型特有参数放入 processor_kwargs=dict(...)
```

#### run_once() — 标准推理流程

```python
def run_once(self, input_str, media_path=""):
    # 1. 确定 media_type
    # 2. process() 分词
    # 3. 检查 token 数不超过限制
    # 4. forward_embed() 嵌入文本 token
    # 5. vit_process() 处理视觉（如果是 VLM）
    # 6. get_rope_index() 计算位置编码
    # 7. forward_prefill() 首次推理
    # 8. 循环 forward_next() 生成后续 token
    # 9. 解码并输出文本
```

#### Unicode 替换字符处理

增量解码时，不完整 UTF-8 序列会产生 `"\ufffd"` 替换字符。缓冲 token 直到完整字符形成：

```python
full_word_tokens = []
while token not in [self.ID_EOS]:
    full_word_tokens.append(token)
    word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
    if "\ufffd" not in word:       # 完整字符已形成
        text += word
        print(word, end="", flush=True)
        full_word_tokens = []
    token = self.model.forward_next(position_ids)
```

⚠️ 源码中使用转义序列 `"\ufffd"` 而非字面 `"�"` 字符，防止文件编码工具破坏该字符。

#### ViT 处理（VLM）

参考 HF 源码的 processor 输出和 `get_image_features()` / `get_video_features()` 流程来实现。
核心逻辑是：把 HF 用 PyTorch 做的 ViT forward 替换为调用 bmodel。

```python
def vit_process_image(self, inputs):
    """处理图片。
    参考 HF 的 get_image_features() 流程：
    1. 从 processor 输出中获取像素数据和 target_sizes
    2. 按 HF 的方式计算 pos_ids、空间索引等
    3. 调用 self.model.forward_vit(...) 替代 HF 的 vision_tower + merger
    4. 将 ViT 输出插入到 text embedding 的正确位置
    """
    ...

def vit_process_video(self, inputs):
    """处理视频。
    参考 HF 的 get_video_features() 流程。
    通常每帧独立过 ViT（无时序建模），与图片处理逻辑类似。
    """
    ...
```

**关键**：processor 输出的字段名和结构因模型而异（如 `pixel_values` vs `pixel_values_videos`、
`target_sizes` vs `image_sizes` 等），以 HF processor 的实际输出为准。

#### get_rope_index() — 位置编码

根据模型的位置编码类型实现：
- **MRoPE**：3D position_ids `[3, batch, seq_len]`，vision token 用 (t, h, w) 三维坐标
- **标准 RoPE**：1D position_ids
- 注意 vision token 后的特殊 token（`</image>`、`<slice>` 等）也要分配 position ID

### 5. 编译和运行

**准备测试文件**：多模态模型需要对应的测试文件（图片/视频/音频）。
从 LLM-TPU 仓库其他 demo 目录拷贝，例如：
```bash
cp ../Qwen3_5/python_demo/test.jpg .
cp ../Qwen3_5/python_demo/test.mp4 .   # 如需视频测试
```

**编译运行**：

```bash
cd python_demo
mkdir build && cd build && cmake .. && make && cp *cpython* .. && cd ..

# 交互模式
python3 pipeline.py -m <model>.bmodel -c ../config

# 单次推理
python3 pipeline.py -m <model>.bmodel -c ../config \
    --prompt "描述这张图片" --media_path test.jpg
```

> ⚠️ 此时精度可能尚未对齐（步骤 4 的工作），生成文本不一定正确。
> 本步骤的目标是**流程跑通**：能编译、能加载 bmodel、能执行推理、能解码输出。
> 精度问题的排查在步骤 4 进行。

**记录环境问题**：运行过程中遇到的环境问题（第三方库版本、缺少依赖、兼容性修复等）
**必须记录到 `<model>_memory.md` 的「关键决策」章节**。这些信息在步骤 5 写 README 的环境准备章节时直接引用。

**记录运行命令**：将最终跑通的编译和运行命令保存到 `<model>_memory.md` 的「关键命令」章节，步骤 4 验证时直接复用。

⚠️⚠️⚠️ **步骤 3 的核心原则：跑通流程，能生成文本就够了。** ⚠️⚠️⚠️

- **不要**在步骤 3 排查精度问题（生成文本乱码、语义不对等）
- **不要**在步骤 3 修改 Converter 或 pipeline 的核心逻辑来尝试修复精度
- **不要**在步骤 3 dump 中间 tensor 做逐层对比

这些全部是步骤 4 的工作。如果步骤 3 发现问题，**只记录到 `<model>_plan.md`**，然后继续推进流程。
步骤 3 完成的标志是：pipeline 能跑完、能输出文本（哪怕内容不对），不是输出正确的文本。

## 完成后：更新 memory

更新 `<work_dir>/<model>_memory.md` 底部「移植进度备忘」章节：
- **关键命令**：填入 cmake 编译命令和 `pipeline.py` 运行命令
- **关键决策**：填入环境问题及解决方案（第三方库版本、兼容性修复等）
- **当前进度**：勾选「步骤 3 Pipeline 跑通」

## 完成标准

- [ ] CMakeLists.txt 编译成功
- [ ] `chat.cpython*.so` 生成
- [ ] 纯文本推理流程跑通（能输出文本，精度待步骤 4 验证）
- [ ] 图片推理流程跑通（VLM）
- [ ] 视频推理流程跑通（如支持）
- [ ] Unicode 替换字符 `"\ufffd"` 处理正确
- [ ] CLI 参数功能正常
- [ ] pipeline 不依赖源模型仓库（仅靠 bmodel + config 即可跑通，不加载 safetensors）
- [ ] 调试过程中遇到的问题已追加到 `<model>_plan.md` 的调试记录章节
- [ ] 用户确认可以进入步骤 4
