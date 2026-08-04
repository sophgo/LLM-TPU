# Step 2: Converter Implementation

> **文件落点**：per-model 文件在 `<work_dir>` = `<repo_root>/models/<model>/tmp/`（`<repo_root>` 为用户指定的 LLM-TPU 仓库路径） 下（见 SKILL.md）。
>
> **前置条件**：步骤 1 完成，`<work_dir>/<model>_plan.md` 已确认。
> **产物**：`<Model>Converter.py` + 编译成功的 bmodel

## 前置：恢复上下文

Read `<work_dir>/<model>_memory.md`，重点看底部「移植进度备忘」章节，从中恢复 repo 路径、环境信息、关键决策和当前进度，再继续下面的步骤。

## 目标

在 tpu-mlir 仓库实现 Converter，将 HF 权重编译为 bmodel。
按 `<model>_plan.md` 中确定的适配顺序执行：**文本 → 视觉（如有）→ 语音（如有）**。
如果某个模态的架构可继承，跳过该模态的实现。

## 执行步骤

### 1. 注册模型

**文件**：`python/tools/llm_convert.py`

在 `LLM_CONVERTERS` 字典中添加条目：

```python
(("<model_type>", ), "llm.<Model>Converter", "<Model>Converter", {
    "default_max_shape": (H, W),    # 从适配计划取
    "pixel_multiple": N,             # 从适配计划取
    "force_dynamic": True/False,     # 从模板的编译模式选择取，非架构特性决定
}),
```

**文件**：`python/llm/__init__.py`

添加导出：
```python
from .<Model>Converter import <Model>Converter
```

**文件**：`python/llm/LlmInfo.py`

仅当文本架构需要新的权重路径映射时修改。可继承时复用已有 `*_INFO`。

### 2. 文本模态

#### 可继承（常见）

```python
from .<BaseConverter> import <BaseConverter>

class <Model>Converter(<BaseConverter>):
    def __init__(self, args, config, loader=None):
        super().__init__(args, config, loader)
```

文本部分零代码，由基类处理。编译时自动继承文本模型的 MLIR 生成。

#### 不可继承（工作量大）

```python
from .LlmConverter import LlmConverter

class <Model>Converter(LlmConverter):
    def __init__(self, args, config, loader=None):
        self.model_info = <NEW>_INFO  # 需要在 LlmInfo.py 新建
        super().__init__(args, config, loader)

    # 需要实现文本模型的全部 MLIR 生成方法
    # 参考已有 Converter 的实现
```

⚠️ 文本不可继承时，文本 Converter 工作量最大。**在步骤 2 内先实现并编译通过文本 Converter**（验证 shape-infer / 算子 lowering / 内存规划无误；语义精度到步骤 4 才验证），再按 **文本 → 视觉 → 语音** 的顺序继续步骤 2 的视觉/语音 Converter。
**不要**跳到步骤 3/4 先把文本 pipeline/精度跑通再回来——各模态 Converter 都在步骤 2 完成，再整体进入步骤 3。详见 1. analyze 的「多模态适配顺序」。

### 3. 视觉模态（如有）

如果适配计划中标注视觉可继承，跳过本节。

#### init_vconfig — 设置视觉参数

```python
def init_vconfig(self):
    """覆写：设置视觉参数。被父类 __init__ 调用。"""
    self.do_vit = True
    self.dynamic = True
    self.vit_path = "<hf_vision_prefix>"
    self.patch_size = N
    self.embed_dim = N
    self.vnum_heads = N
    self.vhead_dim = self.embed_dim // self.vnum_heads
    self.vit_depth = N
    self.intermediate_size = N
    self.spatial_merge_size = N
    # ... 其他模型特有参数
```

#### gen_vit_mlir — 构建 ViT MLIR 图

按 `<model>_plan.md` 的算子映射表，逐层构建：

```
Patch Embedding → Position Embedding → Encoder Layers × N → Post-Norm → Merger
```

**Patch Embedding — 直接用 ConvOp**：

MLIR 的 `top.ConvOp` 之前的精度 bug 已修复，**直接用它实现 Conv2d patch embedding 即可，不再需要 im2col 拆解**。
参考 `Llama3_2VConverter` / `InternVL3Converter` / `JanusConverter` 的写法：

```python
weight_op = vit_mlir.create_weight_op(
    patch_embed + ".weight", [hidden, 3, patch, patch])   # [O, C, kH, kW]
conv_op = top.ConvOp(T([1, hidden, H // patch, W // patch]),
                     in_op,
                     weight_op,
                     vit_mlir.none_op,                       # bias（无则 none_op）
                     kernel_shape=[patch, patch],
                     strides=[patch, patch],
                     pads=[0, 0, 0, 0],
                     dilations=[1, 1],
                     loc=L(patch_embed)).output
# [1, hidden, H', W'] → Reshape → [1, H'*W', hidden]
```

> 旧的 im2col 写法（Reshape+Permute+MatMul）仍可用，但 ConvOp 已是首选方案，
> 只有在 ConvOp 不满足某模型特殊需求时再考虑 im2col。

**Vision Block — 标准结构**：

```python
residual = hidden
hidden = layer_norm(hidden)           # ViT 通常用 LayerNorm
q, k, v = linear(hidden), linear(hidden), linear(hidden)
q, k, v = reshape(q), reshape(k), reshape(v)  # [1, -1, n_heads, d_head]
attn = FAttentionOp(q, k, v, scale=d_head**-0.5, ...)
attn = reshape(attn, [1, -1, D])
out = linear(attn)
hidden = residual + out

residual = hidden
hidden = layer_norm(hidden)
hidden = linear(hidden)               # D → D_ff
hidden = activate(hidden, GELU_xxx)   # 检查是 GELU 还是 GELU(tanh)
hidden = linear(hidden)               # D_ff → D
hidden = residual + hidden
```

**Merger — 空间降采样**：

```python
# 2×2 空间分组：GatherOp(reorder_index) + ReshapeOp
# MLP: LayerNorm → Linear → GELU → Linear
```

**动态 Shape 注意事项**：

如果模型处理变长输入（多 slice、多帧），空间索引必须是 **runtime input**：

```python
# 编译期 weight（固定 shape）→ 只能处理固定尺寸
# 运行时 input（动态 shape）→ 可处理任意 slice 数量和尺寸
```

所有涉及动态维度的 `ReshapeOp` 使用 `-1`：
```python
reshape_op(op, [1, -1, n_heads, d_head])  # ✓ 正确
reshape_op(op, [1, 4624, n_heads, d_head]) # ✗ 硬编码
```

⚠️ **Shape 注意**：只有 `ReshapeOp` 的 `shape` 参数支持 `-1` 动态维度。
其他算子的 output shape 必须是确定的（具体数值），不支持 `-1`。
`ReshapeOp` 自身的 output type 也必须是确定的（由上游算子推导或显式指定）。

索引数据存为 F32（MLIR WeightOp 要求），或作为 runtime input 传入。

### 4. 语音模态（如有）

如果适配计划中标注语音可继承，跳过本节。

语音编码器的适配流程与视觉类似：
- `init_vconfig()` 中设置语音参数（`do_audio = True` 等）
- 实现 `gen_audio_mlir()`：特征提取 → 编码器层 → 投影
- 实现 `compile_audio()`：编译 audio bmodel

具体结构参考 `<model>_plan.md` 中的语音算子映射表。

### 5. 处理模型特异性

根据步骤 1 中用户确认的特异性处理方案，在编译前实现对应的特殊逻辑。

**常见特异性处理模式**：

| 特异性 | 处理方式 |
|--------|---------|
| 多模式（如不同降采样率） | `gen_vit_mlir()` 分发给多个 `_build_vit(mode)` |
| Window attention | GatherOp(window_index) → FAttentionOp(batch=num_windows) → GatherOp(reverse_index) |
| 动态索引 | 索引从 weight 改为 input，MLIR 输入列表增加对应 tensor |
| 混合精度 | 特定层用 float 编译（AWQ 排除列表已自动处理） |

### 6. Converter 常见踩坑

实现过程中高频踩到的坑（来自实际适配记录）：

- **`init_vconfig()` 拿不到 `args`**：`init_vconfig()` 被父类 `__init__` 调用，但 `args` 不是它的参数。需要在 `__init__` 里先把要用到的存到 `self`（如 `self.max_pixels = args.max_pixels`），`init_vconfig()` 再读 `self.*`。
- **`save_weights()` 必须在 `save_mlir_module()` 之前**：debug 模式下 `save_mlir_module()` 会立即做 shape-infer，需要权重文件已存在，否则 crash 报 `*.npz doesn't exist`。
- **Linear 权重尽量用 `set_linear_weight` 存，不要 `self.model.read` 后手动塞 `weights_dict`**：`set_linear_weight(path, weights_dict)` 会自动把 HF 的 `(out, in)` 权重转置成 `(in, out)`，这是 `self.linear()` / `MatMulOp` 期望的布局。手动 `self.model.read` 后直接存入 `weights_dict` 会漏掉转置。
- **只有 `ReshapeOp` 的 `shape` 参数支持 `-1`**：`SliceOp`/`MatMulOp` 等的 output shape 必须是确定数值。动态维度下的切片（如 RoPE 的 rotate_half）不要用 `SliceOp` + 静态 ends——runtime 实际 seq 比编译期小会越界，改用原生 `top.RopeOp`（动态安全）或 `GatherOp`。
- **不要为模型特异需求改共享基类**：如需让某个子网络（如 ViT）强制动态而文本保持默认，在子类 Converter 覆盖 `compile_vit()` 附加 `--dynamic`，**不要改 `LlmConverter.submit_deploy_task` 的门控条件**（会污染所有 LLM 模型，且与 `use_small_mask()` 等其他 `self.dynamic` 分支状态不一致）。
- **BF16 模型量化 dtype 推导**：模型 `torch_dtype=bfloat16` + 4bit 时，框架推导出 `w4bf16` 并强制覆盖用户指定的 `w4f16`（打印 warning）。编译时直接用 `-q w4bf16`。
- **pipeline 不应依赖多 GB safetensors**：可预计算的权重（如位置编码）尽量在 Converter 里算好烘焙进 bmodel，或导出一个小 npz 到 config/，不要让 pipeline 运行时去加载原始权重文件。同理，与输入无关的**常量 runtime input**（如固定分辨率下的窗口注意力 mask、几何索引表等）在静态编译下应改为 bmodel 权重（`create_weight_op`），减少运行时分发体积和 pipeline 复杂度。
- **MatMulSlice lowering 不支持 stride slice**：tpu-mlir 的 MatMulSlice 分解把 stride-2 的 SliceOp（如 `w13[..., 0::2]` 交错切分）退化为连续 slice，导致 gate/up 等交错布局静默回退到错误的前半/后半切分。**修法**：在 Converter 侧**预 deinterleave 权重**——读权重后拆成两个连续权重（如 `w_gate = w13[:, 0::2]`、`w_up = w13[:, 1::2]`），用两个独立 matmul 替代"融合 matmul + stride slice"。
- **多结果 op 的 loc 必须传名字列表**：如果一个 op 有多个结果（如 `FAttentionLseOp` 返回 `(output, lse)`），`loc=` 必须传**名字列表**（如 `loc=self.get_loc(["name.output", "name.lse"], ...)`）。单 FusedLoc 会导致 `getLoc(result_1)` 越界回退到 op loc，两结果同名 → test_mlir 框架 dict 覆盖，第二个结果的数据丢失。

### 7. 编译验证

```bash
llm_convert.py -m /workspace/<weights> -s <seq_len> \
    --max_input_length <input_len> -q <quantize> -c bm1684x \
    --max_pixels <H>,<W> -o <output_dir> --debug
```

> `--debug` 保留 npz 权重文件，步骤 4 用 model_runner 验证时直接可用，**省得步骤 4 再重新编译一次**。步骤 5 写 README 时把 `--debug` 去掉（发布的编译命令不需要保留 npz）。

⚠️ **必须编译出完整的 bmodel**（包含所有模态的网络），不要用 `--dry_run` 或 `--only_mlir` 等只生成中间产物的选项。
完整编译才能验证 shape-infer、算子 lowering、内存规划等全链路是否正确。

⚠️ **遇到算子报错，先查算子定义再改代码**：编译时若报某算子的 shape-infer / 属性 / 签名错误，**先去查该算子的定义**，搞清楚 operands / attributes / output shape 的约定：
- Top dialect：`include/tpu_mlir/Dialect/Top/IR/TopOps.td`
- Tpu dialect：`include/tpu_mlir/Dialect/Tpu/IR/TpuOps.td`
- Python 接口签名：编译后的 `top.<Op>Op(...)` 参数列表
- 已有用法：`grep -rn "<OpName>" python/llm/`

**看懂算子契约后，再回头看 Converter 里这处调用怎么改**——不要在不懂算子定义的情况下盲改参数。若查完发现是算子本身缺失（不是用法错），记录到 `<model>_plan.md` 的待优化章节，先用 workaround 方案绕过。

⚠️ **修改 tpu-mlir 的 cpp / ppl 源码后**：必须先回到 tpu-mlir 根目录执行 `./build.sh` 重新编译后端，改动才会生效。只改 Python 层（Converter）不需要 build.sh。

⚠️ **重新编译时注意**：`llm_convert.py` 检测到输出目录中已有对应 bmodel 时会跳过不重新编译。
如果只修改了某个模块（如 ViT），需要先删除对应的 bmodel 再重新执行：

```bash
# 删除整个输出目录重新编译（最干净）
rm -rf <output_dir> && llm_convert.py ...

# 或只删除修改过的模块的 bmodel（更快）
rm <output_dir>/vit*.bmodel && llm_convert.py ...
```

**记录到 memory**：将最终使用的编译命令保存到 `<model>_memory.md` 的「关键命令」章节，步骤 4 验证和步骤 5 写 README 时直接引用。

**检查**：
- shape-infer 通过（MLIR 结构正确）
- bmodel 文件生成成功
- 用 `model_tool --info <model>.bmodel` 查看 bmodel 的输入输出和对应 shape
- 如果有 ViT / Audio bmodel，检查其输入/输出 shape 是否与预期一致

## 完成后：更新 memory

更新 `<work_dir>/<model>_memory.md` 底部「移植进度备忘」章节：
- **关键命令**：填入最终跑通的 `llm_convert.py` 编译命令（含各目标芯片的参数）
- **关键决策**：填入编译模式选择（静态/动态）、量化方案等
- **当前进度**：勾选「步骤 2 Converter 编译」

## 完成标准

- [ ] 模型已注册（llm_convert.py + __init__.py）
- [ ] Converter 文件已创建，继承关系正确
- [ ] 各模态按适配计划顺序实现（可继承的已跳过）
- [ ] 模型特异性已按方案处理
- [ ] bmodel 编译成功（shape-infer 通过）
- [ ] 编译和调试过程中遇到的问题已追加到 `<model>_plan.md` 的调试记录章节
- [ ] 用户确认可以进入步骤 3
