# Step 4: Precision Verification

> **文件落点**：per-model 文件在 `<work_dir>` = `<repo_root>/models/<model>/tmp/`（`<repo_root>` 为用户指定的 LLM-TPU 仓库路径） 下（见 SKILL.md）。
>
> **前置条件**：步骤 3 完成，端到端推理能跑通（输出可能不正确）。
> **产物**：精度对比报告，确认 bmodel 输出与 HF 参考一致。
> **依赖**：CUDA 环境（模板中填写），用于运行 HF 参考模型获取标准输入/输出。

## 前置：恢复上下文

Read `<work_dir>/<model>_memory.md`，重点看底部「移植进度备忘」章节，从中恢复 repo 路径、环境信息、关键决策和当前进度，再继续下面的步骤。

## 目标

验证 bmodel 的计算精度，定位并修复精度问题。
**这是整个适配过程中最繁琐的步骤。** 步骤 1 的架构分析和算子映射做得越细致，这里的排查就越轻松。

---

## Part 1: 验证流程

### 总体顺序

三个维度的顺序：

**维度 1 — 模态顺序**：文本 → 视觉（如有）→ 语音（如有）

- 可继承的模态：只做 a + b（对齐输入输出流程）
- 新实现的模态：做 a ~ f 全链路

**维度 2 — 编译模式顺序**（根据模板中的选择）：

```
1. 静态 bmodel 验证（如果编译了静态版本）
2. 动态 full shape 验证（图片 Resize / 文本 padding 到 max shape）
3. 动态 small shape 验证（实际输入不做处理，自然 shape）
```

如果模板选了“仅静态”，只做 1；选了“强制动态”，只做 2+3；选了“静态+动态”，做 1+2+3。

**维度 3 — 设备顺序**：主设备（通常 BM1684X PCIe）的精度用上面的 a~f op 级流程深挖通过后，扩展到其他目标芯片/形态（按模板多选：BM1684X SoC / BM1688 SoC）。排查策略见下方「多设备精度扩展」。

### 每个模态的验证链路

```
a. 确保输入与 HF 相同
   → 用 dump 工具（见 §A）在 chat.cpp 中保存 bmodel 实际接收到的输入
   → 与 HF 标准输入（见 §E 获取）对比，确认一致

b. 对比最终输出
   → bmodel 输出 vs HF 标准输出，计算 cosine similarity
   → 达标（见下方精度标准）→ 该模态通过 ✅
   → 不达标 → 进入 c

c. 获取中间算子结果
   → 静态模型优先用 model_runner.py（见 §B）
   → 动态模型：先在 Converter 加临时输出（见 §F）把目标中间算子变成 bmodel 输出，再用 dump 工具重载 2（见 §A）保存这些输出
   → 同时获取 HF 对应层的中间结果（见 §E）

d. 逐层对比，定位第一个精度下降的算子
   → 逐层计算 cosine，找到第一个明显下降的位置
   → 确认是哪个 MLIR 算子出的问题

e. 排查可疑算子
   → 先对照 HF 源码，检查 Converter 中该算子的实现是否与 HF 一致
     （参数、shape、计算顺序、激活函数类型等）
   → 如果不一致 → 修复 Converter 实现
   → 如果确认一致 → 用 test_mlir.py（见 §D）写最小测试，验证算子本身是否有精度问题

f. 修复 → 回到 step b 回归验证
   → 重新编译 bmodel，重新跑对比（⚠️ 若修复涉及 tpu-mlir 的 cpp/ppl 源码改动，须先在 tpu-mlir 根目录执行 `./build.sh`）
   → 确认修复有效且未引入新问题
```

> ⚠️ **每修复一个算子就算一个独立的问题**。修复后回到 step b 重新对比，
> 如果精度仍然不达标，说明**后面还有新的问题**，而不是上一个问题没解决完。
> 不要反复修改同一个算子，要重新从 step c 定位下一个出问题的算子。

> ⚠️ **隔离定位原则**：对可疑算子，**同时对比它的输入和输出**（不只是输出）。
> 若**输入已偏离** HF（cosine 低），说明问题在**上游**，本算子无辜——往上查，不要反复改本算子。
> 只有当输入对齐、输出偏离时，才是本算子的问题。
> 典型：RoPE 输出 cosine 低，先查 RoPE 输入（QKV 投影结果）——若输入已偏离，RoPE 本身不背锅
> （可用 test_mlir 单独验证算子精度排除嫌疑，再往上逐层定位真根因）。

> ⚠️ **输入审计**：进入 step c 逐层排查之前，先确认 step a 的输入**严格一致**——很多看似精度漂移的问题实际是输入不一致造成的：
> - **prompt 构造**：用 `tokenizer.encode()` 对比 TPU prompt 和 HF prompt 的 token ID 序列，哪怕多一个 `\n` 或少一个空格，tokenize 后差 1~N 个 token → prefill 位置编码全偏移、KV cache 长度不一致，后续 decode 每步 hidden state 都错位——看起来像精度漂移，实际是输入不对
> - **图片文件**：md5 校验确认同一文件，预处理一致
> - **prefill token 数**：如果 TPU 和 HF 的 prefill token 数不同，先排查 prompt 构造，不要急着查精度

### 精度判断标准（step b 使用）

| Cosine Similarity | 判定 | 说明 |
|-------------------|------|------|
| > 0.9999 | ✅ 完美 | 无问题 |
| > 0.999 | ✅ 优秀 | 轻微量化噪声，可接受 |
| > 0.99 | ⚠️ 可接受 | 检查下游文本质量是否受影响 |
| > 0.95 | ⚠️ 需关注 | 可能有算子精度问题 |
| < 0.95 | ❌ 有问题 | 需要逐层排查（进入 step c） |
| NaN | ❌ 严重 | 数值溢出，检查 F16 范围或量化配置 |

### 常见精度问题速查

| 现象 | 可能原因 | 排查方向 |
|------|---------|---------|
| ViT cosine 低（< 0.95） | 算子映射错误 | 逐层 dump，定位第一个 cosine 下降的层 |
| Full shape 正常，small shape 低 | 动态 shape 编译问题 | 检查算子在小 shape 下是否正确 |
| 文本 block 0 cosine 就低 | embed 拼接错误 | 检查 ViT 输出插入位置、position_ids |
| 文本 block K 突然 NaN | F16 溢出 | 改用 BF16 编译，或检查 AWQ 量化 |
| 文本 cosine 缓慢下降到 0.9x | 正常量化累积 | 检查端到端文本质量是否可接受 |
| 多 slice cosine 低（~0.7） | padding 影响 attention | 改为逐 slice 独立处理 |
| 生成全乱码 | position_ids 错误 | 检查 shape、值域、MRoPE section |
| 生成重复/空 | ViT 输出全零或 NaN | dump ViT 输出检查值域 |
| LayerNorm cosine 低（bf16+动态） | BM1684X bf16 动态 LN kernel 用不稳定 `var=E[x²]-E[x]²`，输入有 per-position DC（小均值，如 patch embedding 直接入 LN）时灾难性抵消 | LN 前减 per-position 均值（ReduceMean+Sub）workaround，见下方专节 |
| debug 版精度劣于生产版 | 额外 return_ops 强制中间结果写回 global memory，切断 layer-group 融合，多余 bf16 截断累积 | 定位完必须删 debug 输出重编译再验证，debug 精度不代表生产精度 |

#### BF16 动态 LayerNorm 精度问题（BM1684X）

> ⚠️ **临时方案**：这是 BM1684X bf16+动态 LayerNorm 后端 kernel 的 bug，会上报后端修复。
> **后端修复后本 workaround 即可移除**，无需长期保留。修复前踩到的模型按此处理。

BM1684X 的 bf16+动态 LayerNorm kernel 用**不稳定方差公式** `var = E[x²] - E[x]²`。当 LN 输入有非零
per-position 均值（DC）时，`E[x²]` 与 `E[x]²` 相减灾难性抵消 → var 算错 → 输出坏。零 DC 输入不触发；
静态 bf16 / f32 动态用稳定公式 `E[(x-mean)²]`，不受影响。

**触发条件**：LN 输入幅度小、有 per-position DC——典型是 **patch embedding 直接进 LayerNorm**（positional emb
+ patch DC 使 per-pos 均值大）。多数模型 LN 输入是残差累加后的 hidden state（DC 不显著），不会踩到。

**判定**：隔离单 op 测（test_mlir，真实 input+weight，bf16+动态）——若 `cos(bmodel, numpy正确LN)` 远低于
静态 bf16（如 0.76 vs 0.9999），且 per-position 全部均匀偏低（非 padding 分布），即此 bug。

**临时 Workaround**（Converter 侧，仅 `chip=="bm1684x"` 时）：LN 前先减 per-position 均值再过 LN——
数学上 `LN(x-mean) == LN(x)`（LN 对预减均值不变，输出不变），但 LN 输入变零 DC → kernel 不抵消 → var 准。

```python
def vit_layer_norm(self, op, eps, ...):
    if self.chip == "bm1684x":   # 仅 BM1684X 需要，后端修复后删掉这个分支
        mean = top.ReduceOp(..., op, mode="ReduceMean", axes=[-2], keepdims=True).output
        op = top.SubOp(..., [op, mean]).output   # 去 per-position DC
    return self.layer_norm(op, eps, ...)          # 复用原有 LayerNormOp
```

ReduceMean 走单独 reduction kernel（f32 累加），bf16 动态下精度够。不动共享 `LlmConverter.layer_norm`，
只在本模型 ViT 的 norm 处用。BM1688/SG2380 的 bf16 LN 会降到 f32，不需要此 workaround。

### 多设备精度扩展

主设备（通常 BM1684X PCIe）精度通过后，对每个其他目标芯片/形态（按模板多选）：

1. 为该芯片编译 bmodel（MLIR 环境）
2. 在该芯片的设备上跑 pipeline，用 dump 工具（§A 重载 2，动态必须）保存输出
3. cosine vs HF 标准输出，不达标 → 按下面的**设备过渡排查策略**定位

**设备过渡排查策略**（看是从主设备过渡到哪种设备）：

- **同芯片换形态（BM1684X PCIe → BM1684X SoC）**：bmodel 内部**不应**出问题（同一颗芯片，kernel 相同），优先怀疑 **pipeline**（SoC 端代码路径、内存布局、host 侧预处理等差异）。用 dump 工具挨个子 net 对比，逐一对齐 PCIe 上的输出——第一个对不上的子 net 就是问题所在。
- **换芯片（BM1684X → BM1688）**：bmodel 内部**可能**出问题（不同芯片，bf16 等 kernel 行为可能不同）。先用 dump 工具对比各子 net；若定位不到，再给 Converter 加中间输出（§F）定位到具体算子，然后用 test_mlir（§D）分别在该算子上测 bm1684x 和 bm1688，看两颗芯片的算子行为差异。

> ⚠️ 多设备排查依赖 dump 工具和 Converter 临时输出——这些到步骤 5 才清理，所以**多设备精度验证必须在步骤 5 清理之前完成**，发现问题能直接走上面的 op 级流程，不用临时把工具加回来。

**SoC 环境常见踩坑**（多设备跑起来时按需排查）：
- **python 版本**：部分设备默认 `python3`=3.8，需用 `python3.10`，编译时 `cmake -DPython_EXECUTABLE=$(which python3.10) ..`
- **pybind11 路径**：CMakeLists 硬编码 `pybind11_DIR` 路径，需 `pip install --user pybind11`（apt 装的不落在该路径）
- **aarch64 缺 wheel**：某些包（如 `decord`）aarch64 无 PyPI wheel，需创建 stub（未用到的方法 raise）或源码编译
- **numpy 版本**：装 `opencv` 可能升 numpy 到 2.x 与 torch 不兼容，需手动降回 1.26.x
- **processor 依赖**：trust_remote_code processor 的 `check_imports` 可能强制顶层 import 的包全部安装，即使本 demo 不用

---

## Part 2: 工具用法

### §A. dump_net_to_file — 保存 bmodel 运行时输入输出

**来源**：`LLM-TPU/support/debug/` 目录，使用前需拷贝 `utils.h` 和 `cnpy.cpp`/`cnpy.h` 到 `python_demo/`，
CMakeLists.txt 中添加 `cnpy.cpp` 并链接 `z`。参考该目录的 README。

**适用环境**：TPU 运行环境

**两个重载**：

```cpp
// 重载 1：不传 tensor，使用 net->stages[0] 的 max shape
// ✓ 静态模型可用（max shape == actual shape）
// ✗ 动态模型不可用（max shape != actual shape，dump 结果不对）
void dump_net_to_file(bm_handle_t &bm_handle, const bm_net_info_t *net,
                      const std::string &filename);

// 重载 2：传入实际 tensor，使用 tensor 自身的 runtime shape
// ✓ 动态模型必须用这个
// ✓ 静态模型也可以用
void dump_net_to_file(bm_handle_t &bm_handle, const bm_net_info_t *net,
                      const std::vector<bm_tensor_t> &in_tensors,
                      const std::vector<bm_tensor_t> &out_tensors,
                      const std::string &filename);
```

⚠️ **动态模型必须用重载 2。** 重载 1 读取的是编译时 max shape，动态模型实际 shape 比它小，dump 出的数据 shape 不对。

**使用方式**：先在 `chat.cpp` 顶部添加 `#include "utils.h"`（utils.h 从 `support/debug/` 拷贝过来），
然后在 `net_launch()` 之后添加调用：

```cpp
#include "utils.h"

// 在 chat.cpp 的 forward_vit / forward_first 中，net_launch 之后加：
dump_net_to_file(bm_handle, net_vit, in_tensors, out_tensors, "vit_io.npz");

// 也可以只 dump 输入或输出：
dump_net_input_to_file(bm_handle, net_vit, in_tensors, "vit_input.npz");
dump_net_output_to_file(bm_handle, net_vit, out_tensors, "vit_output.npz");
```

**输出格式**：npz 文件，key 为 tensor name，value 为转为 float32 的数据。

### §B. model_runner.py — 跑 MLIR / 静态 bmodel

**适用条件**：MLIR 编译环境，静态模型

**前置条件**：用 `llm_convert.py` 编译时必须加 `--debug` 参数，才会**保留 npz 权重文件**——model_runner 跑 `.mlir` 时要加载这些权重，不加 `--debug` 权重会被清理掉、`.mlir` 跑不起来。注意 `--debug` 保留的是 npz 权重，不是 `.mlir` 文件本身。
运行时需要进入 model 对应的输出文件夹（否则找不到权重文件）：

```bash
cd <output_dir>/<model_name>
model_runner.py --model <name>.mlir --input hf_input.npz --output mlir_output.npz
```

**基本用法**：

```bash
# 跑 MLIR 文件，验证 MLIR 层结果
model_runner.py --model <name>.mlir --input hf_input.npz --output mlir_output.npz

# 跑静态 bmodel 文件，验证编译后结果
model_runner.py --model <name>.bmodel --input hf_input.npz --output bmodel_output.npz

# 跑 MLIR 文件，dump 所有中间算子结果
model_runner.py --model <name>.mlir --input hf_input.npz --output mlir_output.npz --dump_all_tensors
```

**注意**：`--dump_all_tensors` 只能 dump `.mlir` 文件的中间算子，**不能** dump bmodel 的中间算子。

**验证步骤**：
1. 先分别用 model_runner 跑 `.mlir` 和 `.bmodel`，对比两者的最终输出是否一致
2. 如果一致 → mlir 的 `--dump_all_tensors` 结果可作为 bmodel 中间算子的等价参考
3. 如果不一致 → 说明编译过程（lowering/optimize）引入了问题，需要用 §F（Converter 加临时输出）来定位

### §C. model_tool --info — 查看 bmodel 信息

**适用环境**：TPU 环境 或 MLIR 环境

```bash
model_tool --info <model>.bmodel
```

输出 bmodel 中所有网络的名称、输入/输出 tensor 的 name、shape、dtype。
用于快速确认编译结果是否符合预期。

### §D. test_mlir.py — 算子级最小测试

**适用环境**：MLIR 编译环境（`tpu-mlir/python/test/` 目录下）

**何时使用**：逐层对比已定位到可疑算子，且确认 Converter 实现与 HF 一致（不是映射错误），
需要验证算子本身在 TPU 上是否有精度问题。

**用法**：

```bash
cd python/test

# 运行指定测试用例
python test_mlir.py --case <case_name> --chip bm1684x --mode bf16 --debug

# 动态算子需要加 --dynamic
python test_mlir.py --case <case_name> --chip bm1684x --mode bf16 --debug --dynamic
```

**编写测试用例**：直接在 `test_mlir.py` 文件中编写，用 HF 的标准输入/输出数据构建 MLIR 图：

```python
def test_<case_name>(self, case_name):
    """针对可疑算子的最小测试。"""
    # 1. 构造输入（使用 HF 实际数据，非随机）
    # 2. 创建 MLIR 图（仅包含该算子）
    # 3. 部署为 bmodel 并对比
```

测试用例命名应通用（如 `conv2d_non_overlapping`、`matmul_reshape_permute`），不绑定特定模型。

### §E. HF hook dump — 获取 HF 参考数据

**适用环境**：CUDA 环境

**前置条件**：CUDA 环境需配置好 HF 运行环境（transformers 版本、模型权重路径等）。
**配置过程记录到 `<model>_plan.md`，最终能用的命令保存到 `<model>_memory.md` 的「关键命令」章节**，方便后续复用。

> ⚠️⚠️ **先研究清楚环境，再动手装——这是 CUDA 环境最浪费时间的一环**。在 CUDA 设备上跑 HF 模型，环境（尤其 torch 版本）极易踩坑：torch 安装很慢，装完发现版本不对又得卸了重装，往往反复好几次才跑起来，半天就没了。**动手 `pip install` 之前，先把整个环境组合定下来、确认它能跑该模型，再开始装**：
> 1. 查 HF 模型卡 README 推荐的 `python` / `transformers` / `torch` 版本组合（模型卡「Usage / 环境配置」段通常有写）
> 2. `nvidia-smi` 看 driver 版本，确认 torch wheel 的 cuXX 后缀与 driver 兼容
> 3. trust_remote_code 模型还要确认模型特有依赖（如 `decord` / `flash-attn` / `xformers` 等）的版本要求
> 4. **把这组版本记到 `<model>_plan.md`、确认它能跑该模型后，再开始 `pip install`**
> 5. **尽量后台安装**（`pip install ... &` 或后台任务），安装期间继续干别的步骤——别干等这十几到半小时
>
> 不要边装边试——torch 装一次动辄十几分钟到半小时，装错重来几次一天就废了。

**目的**：获取两种数据：
1. **标准输入** — 传给 bmodel 用（确保输入一致）
2. **中间结果** — 与 bmodel 中间算子对比用

> 💡 **首次配好环境后，先按 HF 仓库 README 的示例代码跑通 demo**（确认模型本身能正常加载 / 推理、环境没问题），**再**用 hook 拿中间 tensor。环境没跑通就上 hook，等于在错误的地基上排错——白搭。**跑通后把运行方式（命令 / 脚本 / 环境）记到 `<model>_memory.md` 的「关键命令」章节**——这个 HF 工具后面可能反复用，省得每次重新查怎么跑。

**脚本模板**：

```python
import torch
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(model_path, ...)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 1. 获取标准输入
inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, ...)
np.savez("hf_input.npz",
         pixel_values=inputs.pixel_values.numpy(),
         input_ids=inputs.input_ids.numpy(),
         ...)

# 2. 用 hook 捕获中间 tensor
dumps = {}
_dump_done = False

def make_hook(name):
    def hook(module, input, output):
        if _dump_done:
            return
        if isinstance(output, torch.Tensor):
            dumps[name] = output.detach().cpu().float().numpy()
        elif isinstance(output, tuple):
            dumps[name] = output[0].detach().cpu().float().numpy()
    return hook

for name, module in model.named_modules():
    if any(k in name for k in ["vision_tower", "merger", "layers.0", "layers.1"]):
        module.register_forward_hook(make_hook(name))

# 3. 运行 generate（只取 prefill 阶段的数据）
output = model.generate(**inputs, max_new_tokens=1)
_dump_done = True

# 4. 保存
np.savez("hf_intermediate.npz", **dumps)
```

**注意事项**：
- 用 `_dump_done` flag 防止 decode 阶段覆盖 prefill 数据
- 所有 tensor 转 float32 保存（匹配 dump 工具的输出格式）
- 将 npz 文件 SCP 回 MLIR/TPU 环境进行对比
- 环境配置和使用命令记录到 `<model>_memory.md` 的「关键命令」章节

### §F. Converter 加临时输出 — 获取 bmodel 中间算子结果

**适用条件**：动态和静态模型均可用。静态模型优先用 §B（model_runner.py），§B 不够用时再用此方法。

**何时使用**：
- model_runner.py 跑 mlir 和 bmodel 的最终输出不一致（编译过程有问题）
- 动态模型需要获取 bmodel 内部中间算子结果

**方法**：在 Converter 的 `_build_vit()` 或 `gen_block_mlir()` 中，临时添加中间算子作为额外输出：

> 💡 **首次使用：先列算子、一次加全**。对照 HF 源码列出模型用了哪些算子、哪些可疑（精度敏感 / 量化后易坏 / 不常见 / 新映射的），**一次性把这些算子都加成临时输出**，重新编译一次 bmodel，一次 dump 拿到全部中间结果对比——尽量一步定位到明确有问题的算子。**别每加一个输出就重新编译一次**：每次改 Converter 都要重新编译 bmodel，很费时间。

```python
# 1. MLIRImporter 的输出 shape 列表（第 2 个参数）要为每个临时输出加一项
#    参考 LlmConverter.gen_block_by_length：
#    MLIRImporter(input_shapes, [input_shape, kv_shape, kv_shape,  # 原有输出
#                                 block_0_shape, attn_shape, ...],  # 临时加的
#                 name, ...)
#    列表长度必须 == return_ops 总数，否则 shape-infer 报错
block_mlir = MLIRImporter(input_shapes,
                          [out_shape, block_0_shape, block_6_shape, attn_shape, ...],
                          name, ...)

# 2. return_ops 把临时输出一起 return（验证完后必须删除！）
return_ops = [final_output]
return_ops.extend([block_0_out, block_6_out, attn_output, ...])
block_mlir.create_return_op([new_op] + return_ops)
```

⚠️ **不只是加 `return_ops`**：每加一个临时输出，还要在 `MLIRImporter(input_shapes, [输出shape列表], ...)` 的**第 2 个参数**里加对应 shape，两者数量必须一致，否则 shape-infer 报错。参考 `LlmConverter.gen_block_by_length`：第 2 个参数 `[input_shape, kv_shape, kv_shape]` 与 `create_return_op([new_op] + [k_op, v_op])` 一一对应。

重新编译后 bmodel 会有多个输出，用 dump 工具（§A）或 model_runner（§B）拿到这些中间输出与 HF 对比。

⚠️ **验证完后必须删除临时输出**，步骤 5 清理时确认。

⚠️ **debug 输出会改变精度特性**：每个额外 `return_ops` 都是 `func.return` 的 operand，
编译器必须在该点把 tensor 写回 global memory（强制 bf16 截断），这会**切断 layer-group 融合**、
在多层 encoder 中引入多个额外的 bf16 round-trip 点，累积误差被放大。
**因此 debug 版的精度严格劣于生产版**——定位完 bug 后，必须删掉所有临时输出、重编译、
重新验证最终精度，不要用 debug 版的精度下结论。这也是步骤 5 "先清理 Converter 额外输出 → 重新编译"
必须在最前面执行的原因。

---

## 过程中：持续记录

步骤 4 是迭代最密集的步骤，**每完成一个算子的定位和修复**：
- **plan.md**：追加详细调试记录（dump 分析、cosine 值、根因、修复方案）
- **memory.md**：更新「当前进度」（一行摘要即可）

```markdown
# memory.md 示例：
- [ ] 步骤 4 精度验证 ← 当前：ViT ✅, block 0~3 ✅, block 4 LayerNorm cosine 0.76 定位中
```

## 完成后：更新 memory

更新 `<work_dir>/<model>_memory.md` 底部「移植进度备忘」章节：
- **关键决策**：填入精度问题的根因和修复方案（哪些算子有问题、怎么修的）
- **当前进度**：勾选「步骤 4 精度验证」

## 完成标准

- [ ] 各模态精度达标（按适配计划中的顺序逐个验证）
- [ ] 静态/动态各验证通过（按编译模式要求）
- [ ] 端到端生成文本语义正确
- [ ] 所有精度问题已修复并回归验证
- [ ] 多设备精度验证通过（按模板选的目标芯片/形态：同芯片换形态查 pipeline、换芯片查 bmodel 内部），各芯片最终 bmodel 路径已记入 `<model>_memory.md` 的「关键命令」章节
- [ ] dump 代码和 Converter 中的临时输出已标记为待清理（步骤 5 清理）
- [ ] 调试记录已追加到 `<model>_plan.md`
- [ ] 用户确认精度满足要求，可以进入步骤 5
