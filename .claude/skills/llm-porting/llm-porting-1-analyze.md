# Step 01: Architecture Analysis

> **文件落点**：per-model 文件在 `<work_dir>` = `<repo_root>/models/<model>/tmp/`（`<repo_root>` 为用户指定的 LLM-TPU 仓库路径） 下（见 SKILL.md）。
>
> **前置条件**：`<work_dir>/<model>_memory.md` 已由 SKILL 复制并由用户填写完整，模型权重已下载。
> **产物**：`<work_dir>/<model>_plan.md`（适配计划 — 活文档，贯穿整个适配流程，后续步骤中持续补充记录）

## 前置：恢复上下文

Read `<work_dir>/<model>_memory.md`，重点看底部「移植进度备忘」章节，从中恢复 repo 路径、环境信息、关键决策和当前进度，再继续下面的步骤。

## 目标

读取 HuggingFace 源码，自动完成所有技术分析，输出结构化报告。
**人只需确认分析结果，并决策模型特异性内容的处理方式。**

## 执行步骤

### 1. 读取模型 README

从 HuggingFace 模型仓库读取 README.md，提取以下信息：

- **模型介绍**：参数量、架构组成、支持的模态（步骤 5 写 demo README 时直接引用）
- **输入输出格式**：支持的输入类型（文本/图片/视频/音频）、输出格式
- **示例代码**：processor 的调用方式、message 格式、特殊参数（步骤 3 pipeline 的 `process()` 直接参考）
- **环境配置**：推荐的 transformers 版本、额外依赖（`trust_remote_code`、特殊包等）
- **已知限制**：最大分辨率、最大帧数、特殊约束等

### 2. 读取模型配置

读取 `config.json`，提取：
- `model_type` — 判断文本架构是否匹配已有模型
- 文本配置（`text_config` 或顶层字段）
- 视觉配置（`vision_config` 或等效字段，如有）
- 语音配置（`audio_config` 或等效字段，如有）
- 特殊参数（`partial_rotary_factor`、`tie_word_embeddings`、量化配置等）

### 3. 读取 HF 建模源码

读取 `modeling_<name>.py`（通常在 HF cache 中），按模态分析：

**文本模型**：
- Forward 调用链：`generate()` → `forward()` → 各子模块
- 注意力类型：MHA / GQA / MQA / 混合（linear + full）
- Norm 类型：RMSNorm / ZeroCenteredRMS / LayerNorm
- 激活函数：SiLU / GELU / 其他
- 位置编码：RoPE / MRoPE / 其他
- 特殊机制：gated Q projection、QK-norm、MoE 等

**视觉模型**（如有）：
- ViT 结构：层数、hidden_size、num_heads、intermediate_size
- Patch embedding：Conv2d 参数（kernel_size, stride）
- 降采样方式：几级 2×2 merge、是否有 window attention
- 激活函数：GELU / GELU(tanh) — 注意不同层可能不同
- Norm 类型：通常是 LayerNorm（有 bias），区别于文本模型的 RMSNorm
- 位置编码：可学习 lookup / bucketize / 插值

**语音模型**（如有）：
- 编码器结构：层数、hidden_size、num_heads
- 特征提取：卷积参数、mel 频谱参数
- 降采样/投影方式
- 适配流程与视觉类似（编码 → 投影 → 拼接到文本序列）

### 4. 读取权重文件索引

打开 `model.safetensors.index.json`，列出所有权重路径前缀，建立映射表：

```
| 组件 | HF 权重路径 | Converter 加载路径 |
|------|------------|------------------|
| ... | ... | ... |
```

如果模型有 AWQ/GPTQ 量化，检查 `modules_to_not_convert`，标记哪些模块必须保持 float。

### 5. 确定继承关系

根据分析结果，自动匹配最佳基类：

```
config.json 的 model_type 或文本架构特征
    ↓
匹配已有 Converter 的文本架构
    ↓
找到 → 继承该 Converter（文本部分零代码）
找不到 → 继承 LlmConverter（需实现完整文本 MLIR）
```

视觉和语音部分同理匹配：如果 LLM-TPU 仓库中已有模型的某个模态处理方式相同，pipeline 阶段可直接复用代码。

已有 Converter 文本架构速查（检查 `python/llm/__init__.py` 获取最新列表）：

| 基类 | 文本架构特征 |
|------|------------|
| `Qwen3_5Converter` | mixed linear+full attn, gated Q, QK-norm, ZeroCenteredRMS, MRoPE |
| `Qwen3VLConverter` | standard GQA, MRoPE |
| `Qwen2_5VLConverter` | standard GQA, MRoPE |
| `LlmConverter` (llama) | standard GQA/MHA, RMSNorm |
| `Chatglm3Converter` | ChatGLM 特有 |
| `Phi3Converter` | Phi 特有 |
| `Gemma4Converter` | Gemma 特有（含语音） |

### 6. 确定模型输入

**关键步骤**：搞清楚 processor 输出的 inputs 包含哪些 tensor，决定了 Converter 需要构造哪些输入。

方法：在 CUDA 环境（或本地 CPU）运行 processor，查看输出：

```python
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, ...)

for key, val in inputs.items():
    if hasattr(val, 'shape'):
        print(f"{key}: shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"{key}: {type(val)}")
```

分析每个 tensor：
- **作用**：这个 tensor 在模型 forward 中做什么用？
- **是否固定 shape**：哪个维度随输入变化（通常是 seq_len / num_patches）
- **是否作为 Converter 输入**：
  - `input_ids` → 文本 token，作为 embedding 层 bmodel 的输入
  - `pixel_values` → 视觉像素，作为 ViT bmodel 输入
  - `target_sizes` → 元数据，pipeline 用来计算索引，不进 Converter
  - `position_ids` → 可能进 Converter（如 pos_ids 作为 ViT 输入）
  - `audio_features` → 语音特征，作为 Audio bmodel 输入
  - `attention_mask` → 视模型而定，可能需要作为 bmodel 输入

> ⚠️ 此表是初步分析，步骤 2 实现 Converter 时可能会调整（如增减 attention_mask 等输入）。

输出输入定义表：

```
| 输入名 | 来源 | shape | 不固定维度 | 作为哪个 bmodel 输入 |
|--------|------|-------|-----------|-------------------|
| input_ids | processor | [1, seq_len] | seq_len | embedding 层 |
| pixel_values | processor | [1, 3, P, N*P] | N (patches) | ViT |
| pos_ids | pipeline 计算 | [N] | N | ViT |
| reorder_index | pipeline 计算 | [N_out*4] | N_out | ViT |
| target_sizes | processor | [num_slices, 2] | num_slices | ✗ 元数据 |
```

### 7. 计算 pixel_multiple

基于步骤 5 确定的输入格式和视觉降采样结构计算：

```
pixel_multiple = patch_size × (所有 2×2 merge 的乘积)

示例：
  patch_size=14, 1 级 2×2 → 28
  patch_size=14, 2 级 2×2 → 56
  patch_size=16, 1 级 2×2 → 32
```

验证 `default_max_shape` 能被 `pixel_multiple` 整除。

### 8. 建立算子映射表

将 HF forward 中的每个算子映射到 MLIR 实现。
**每个模态（视觉、语音）分别建立映射表。**

**Reshape 约定**：不固定的维度（通常为 seq_len / num_patches）用 `-1` 表示，例如 `[1, -1, n_heads, d_head]`。

**不常用算子**：如果在已有 Converter 代码中找不到对应的 MLIR 用法，去 tpu-mlir 的算子接口代码中查签名确认参数：
- Top 算子定义：`include/tpu_mlir/Dialect/Top/IR/TopOps.td`
- Python 接口：编译后的 `top.<OpName>Op(...)` 参数列表
- 已有 Converter 中的用法：`grep -r "<OpName>" python/llm/`

**量化权重（AWQ/GPTQ/AutoRound）**：量化后的权重（qweight/qzeros/scales）用 `self.linear` 自动处理，
内部调用 A16MatMul 算子完成反量化+矩阵乘法，Converter 不需要手动拆解量化权重。
- AWQ/GPTQ：标准格式，直接支持
- AutoRound：`packing_format="auto_round:auto_gptq"` 会被框架自动识别为 GPTQ 模式（见 `ModelHandle.py`），
  同样走 `self.linear` → A16MatMul，Converter 无需特殊代码；量化范围外的层（vision/projector/embed/lm_head）保持 BF16

⚠️ **先查已有算子，不要用小算子拼**：MLIR 的 Top dialect 已经实现了大量融合算子。
很多操作 HF 里是用基础算子手写的，但 MLIR 有原生融合实现，**直接调用原生算子即可**，
不要照着 HF 用一堆 MulOp/SliceOp/ConcatOp 去拼。典型例子是 **RoPE**：
- ❌ 错误做法：照 HF 用 `Mul + Slice + MulConst(-1) + Concat + Mul + Add` 拼 `rotate_half`
- ✅ 正确做法：直接用 `top.RopeOp(q, sin, cos, rope_mode="contiguous_halves")`
  （见 `LlmConverter.apply_rotary_pos()`，LlmConverter 里保留的 `rotary_pos()` 是旧的手写实现，已不再使用）

`self.*` 辅助方法（`LlmConverter` 已封装，优先用，别自己裸写算子）：

| 辅助方法 | 对应 MLIR 算子 | 说明 |
|----------|--------------|------|
| self.linear() | MatMulOp / A16MatMulOp(量化) | 含 bias（AddOp）；量化权重自动走 A16MatMul |
| self.mlp() | MlpOp(融合) 或 3×MatMulOp | gate/up/down 融合；非融合时手动拆 |
| self.rms_norm() | RMSNormOp | 注意 ZeroCentered 变体（weight_type 参数） |
| self.layer_norm() | LayerNormOp | ViT 常用（带 bias），区别于文本 RMSNorm |
| self.activate() | SiLU/GELU/Swish/Relu/Sigmoid Op | 按 ActType 分派 |

Top dialect 算子映射表（按功能分组，`top.<OpName>Op` 形式调用）：

```
┌─ 范化 ──────────────────────────────────────────────────────────────┐
│ RMSNorm              → top.RMSNormOp(weight, eps, weight_type)        │
│ LayerNorm            → top.LayerNormOp(weight, bias, eps)              │
│ 手写 RMSNorm(无原生) → ReduceOp + AddConstOp + RsqrtOp + MulConstOp + MulOp │
│                        (仅当算子签名不满足时才走手写路径)              │
├─ 线性 / 矩阵乘 ───────────────────────────────────────────────────────┤
│ Linear               → self.linear() (MatMulOp + AddOp bias)           │
│ 量化 Linear (AWQ/GPTQ)→ self.linear() 自动 A16MatMulOp，无需手拆       │
│ 融合 MLP             → top.MlpOp (gate+up+down+act 融合)              │
│ 手写 MLP             → MatMulOp × 3 + activate                       │
├─ 激活函数 ─────────────────────────────────────────────────────────────┤
│ SiLU / SwiGLU        → top.SiLUOp / self.activate(ActType.SILU)        │
│ GELU                 → top.GELUOp / self.activate(ActType.GELU)       │
│ GELU(tanh 近似)      → self.activate(ActType.GELU_PYTORCH_TANH) ⚠️ 区分 │
│ Swish / Relu / Tanh / Sigmoid → 对应同名 Op                            │
├─ 位置编码 ─────────────────────────────────────────────────────────────┤
│ RoPE                 → top.RopeOp(rope_mode="contiguous_halves") ✅原生 │
│   cos/sin 查表       → top.GatherOp(cos_weight, pos_ids, axis=0)       │
│ MRoPE (3D 位置)      → pipeline 侧算好 pos_ids 传入，Converter 仍用 RopeOp │
├─ 注意力 ───────────────────────────────────────────────────────────────┤
│ Multi-head / GQA Attention → top.FAttentionOp(scale, batch, q_head,   │
│                          kv_head, dim, mq, mk, mask_size, mask_op)     │
│ Attention mask       → get_fattention_mask_op() 生成，prefill/decode 不同 │
├─ 卷积（视觉/语音）─────────────────────────────────────────────────────┤
│ Conv2d (patch_embed)  → top.ConvOp(weight, kernel_shape, strides, pads,  │
│   / Conv1d (音频特征)   dilations, bias) ✅精度 bug 已修复，直接用，别再 im2col │
├─ Shape 操作 ───────────────────────────────────────────────────────────┤
│ view / reshape       → top.ReshapeOp(shape=[1,-1,n_h,d_h]) ✅唯一支持 -1 │
│ permute / transpose  → top.PermuteOp / ReshapeOp 组合                  │
│ concat               → top.ConcatOp(ops, axis=)                       │
│ slice                → top.SliceOp(offset, steps, ends, axes)         │
│ unsqueeze / tile     → top.UnsqueezeOp / top.TileOp                    │
├─ 索引 ─────────────────────────────────────────────────────────────────┤
│ Gather (取行)         → top.GatherOp(weight, index, axis=)             │
│ GatherElements       → top.GatherElementsOp (按元素取)                │
│ ScatterElements      → top.ScatterElementsOp                           │
│ ScatterND            → top.ScatterNDOp                                 │
├─ 逐元素运算 ───────────────────────────────────────────────────────────┤
│ Mul / Add / Div      → top.MulOp / top.AddOp / top.DivOp (两 tensor)   │
│ Mul*const / Add*const→ top.MulConstOp(const_val) / top.AddConstOp     │
│ Rsqrt / Neg          → top.RsqrtOp / MulConstOp(-1.0)                  │
├─ 比较 / 选择 ──────────────────────────────────────────────────────────┤
│ Compare              → top.CompareOp / top.CompareConstOp              │
│ Where (条件选择)      → top.WhereOp                                    │
├─ 归约 / 采样 ──────────────────────────────────────────────────────────┤
│ ReduceMean / ReduceSum → top.ReduceOp(mode=, axes=)                    │
│ TopK                 → top.TopKOp (k=, sorted in lm_head/speculative)  │
│ Softmax              → top.SoftmaxOp(axis=)                            │
│ CumSum               → top.CumSumOp (speculative decoding 用)           │
├─ 权重 / 占位 ──────────────────────────────────────────────────────────┤
│ 常量权重             → mlir_gen.create_weight_op(name, shape, dtype)  │
│ 空 tensor (无 bias)  → mlir_gen.none_op                              │
└───────────────────────────────────────────────────────────────────────┘
```

> 上表覆盖 `LlmConverter.py` 中实际使用的全部 Top 算子。视觉/语音模态如用到额外算子
> （如 Conv2d→`top.ConvOp`、Resample、AvgPool 等），按相同方式在算子接口代码中查签名后补到本表。

**缺失算子处理优先级**：

遇到一个 HF 算子没有直接对应的 MLIR 写法时，按以下顺序排查：

```
1. 查找是否存在对应算子（优先）
   → 先去 Top dialect 确认是否真的"缺失"——很多时候 MLIR 已有原生算子，
     只是 HF 用基础算子手写而已（典型：RoPE、Conv2d）
   → 查 TopOps.td / grep 已有 Converter / 查算子接口签名（见上方「不常用算子」）
   → 找到 → 直接调用原生算子，本算子其实不缺失，结束

2. 等价替换（确认缺失后）
   → 用已有算子组合实现，如某算子没有原生版 → 用 Mul/Add/Reshape 等拼

3. 替换但性能有损（workaround）
   → 功能正确但效率不如原生算子
   → 先跑通全流程，记录到适配计划的「待优化」章节
   → 后续优化替换

4. 开发自定义算子（最后手段）
   → 无等价替换方案，且对性能影响大
   → 需要先开发自定义算子，再继续适配
```

### 9. 识别模型特异性（⚠️ 重点关注）

**这是步骤 1 最重要的输出。** 分析模型中有哪些内容偏离通用流程，需要人做决策。

常见特异性类型：

| 类型 | 示例 | 需要人的决策 |
|------|------|------------|
| 多模式 | 同一模型有多种推理模式（如不同分辨率/降采样率） | 是否都支持，优先级 |
| 动态索引 | 索引依赖运行时输入（如多 slice NaViT） | 改为 runtime input 还是限制使用场景 |
| 精度敏感 | 某些算子在量化后精度大幅下降 | 是否混合精度，哪些层保持 float |
| 特殊 token 结构 | token 序列中有非标准的特殊 token 组合 | pipeline 中如何处理 |
| 新文本架构 | 文本模型无法继承已有 Converter | 评估工作量，确认分阶段策略 |
| 缺失算子 | 模型需要 MLIR 中没有的算子 | 等价替换 / workaround |
| 后处理流程 | 模型有结构化输出（检测框、分割掩码、分类标签等），HF 有完整后处理（crop → resize → sigmoid → NMS → RLE 等） | 分析 HF 源码的后处理流程，在 pipeline 中实现对齐。步骤 4 精度验证依赖正确的后处理——如果只跑通"不崩"的 summary 而不补全后处理，端到端精度验证无法进行 |

**多模态适配顺序**：

后续步骤 2/3/4 中，每个步骤都按 **文本 → 视觉（如有）→ 语音（如有）** 的顺序依次适配：

```
步骤 2: 文本 Converter → 视觉 Converter → 语音 Converter
步骤 3: 文本 pipeline → 视觉 pipeline → 语音 pipeline
步骤 4: 文本精度验证 → 视觉精度验证 → 语音精度验证
```

如果某个模态的架构可以继承已有 Converter，则跳过该模态的 Converter 实现，
pipeline 阶段从 LLM-TPU 仓库对应模型复制代码，精度验证阶段只对齐输入输出流程。

**输出格式**：列出每个特异性，附上建议方案，等用户确认或选择。

### 10. 输出适配计划

生成 `<model>_plan.md`，这是一份**活文档**，在后续步骤中持续补充记录（踩坑、修复、决策等）。

```markdown
# [模型名] 适配计划

## 1. 模型概览
- 参数量、架构组成、支持的模态（文本/视觉/语音）

## 2. 继承决策
- 文本：[基类名] 或 "无匹配，需新建"
- 视觉：[可复用 LLM-TPU 哪个模型的 pipeline 代码] 或 "需新写"
- 语音：[同上] 或 "N/A"

## 3. 模型输入定义
[输入定义表，来自步骤 5]

## 4. 编译参数
- pixel_multiple: [N]
- default_max_shape: [H, W]
- 编译模式: [仅静态 / 静态+动态 / 强制动态]

## 5. 权重路径映射
[完整表格，按模态分组]

## 6. 算子映射表
[按模态分组：文本 / 视觉 / 语音]
[Reshape 动态维度标注 -1]
[缺失算子的处理方案]

## 7. 模型特异性（⚠️ 需用户决策）
[列出每个特异性 + 建议方案 + 用户决策结果]

## 8. 适配顺序
- 步骤 2: 文本 → [视觉] → [语音]
- 步骤 3: 文本 → [视觉] → [语音]
- 步骤 4: 文本 → [视觉] → [语音]
[标注哪些可继承跳过、哪些需要新写]

## 9. 待优化（workaround 记录）
[当前用等价替换但性能有损的算子，后续可优化]

## 10. 调试记录（后续步骤补充）
[步骤 2-4 中遇到的问题和解决方案，持续追加]
```

> 📝 **此文档的价值**：记录完整的适配过程，包括踩过的坑和解决方案。
> 适配完成后，此文档可用于复盘和优化 SKILL 流程。

## 完成后：更新 memory

更新 `<work_dir>/<model>_memory.md` 底部「移植进度备忘」章节：
- **关键决策**：填入继承关系、pixel_multiple、default_max_shape、模型特异性处理方案
- **当前进度**：勾选「步骤 1 架构分析」

## 完成标准

- [ ] `<model>_plan.md` 已生成
- [ ] 继承关系已确定（文本 / 视觉 / 语音各自）
- [ ] 模型输入定义表完整
- [ ] pixel_multiple 和 default_max_shape 已计算验证
- [ ] 权重路径映射完整（按模态分组）
- [ ] 算子映射表完整（含 Reshape -1 标注、缺失算子处理方案）
- [ ] 模型特异性已列出，用户已确认处理方案
- [ ] 适配顺序已明确（哪些模态需要新写，哪些可继承跳过）
- [ ] 用户确认可以进入步骤 2
