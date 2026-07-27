# Step 5: Cleanup and Documentation

> **文件落点**：per-model 文件在 `<work_dir>` = `/workspace/llm/LLM-TPU/models/<model>/tmp/` 下（见 SKILL.md）。
>
> **前置条件**：步骤 4 完成，精度验证通过。
> **产物**：清理后的代码 + README.md

## 目标

清理所有调试代码，编写部署文档。

## 执行步骤

### 1. 清理 debug 代码（按顺序）

⚠️ **清理有顺序，必须按以下步骤依次执行。**

#### 1a. 先清理 Converter 的额外输出

Converter 中步骤 4 临时添加的中间算子输出（`return_ops.extend([...])`）会改变 bmodel 的结构，
必须先删掉才能编译出干净的 bmodel：

- 删除 `gen_block_mlir()` / `_build_vit()` 中的额外 return ops
- 删除所有 `# DEBUG` 注释和 `debug_layer0` 等调试字典
- **保留** 正常的注释和文档字符串

#### 1b. 重新编译不带中间算子的 bmodel

删除输出目录中的旧 bmodel，用 `llm_convert.py` 重新编译（命令见 `<model>_plan.md`）：

```bash
rm -rf <output_dir>  # 或只删除修改过的模块 bmodel
llm_convert.py ...   # 用 plan 中记录的命令
```

编译后用 `model_tool --info` 确认 bmodel 的输出只有最终结果，没有多余的中间算子输出。

#### 1c. 重新验证最终输出（视情况）

> 参考 `<model>_plan.md` 调试记录末尾：步骤 4 的 bmodel 如果带有中间算子输出，
> 可能引入累积误差。现在 bmodel 已经清理干净，需要确认清理后的精度。

- **如果步骤 4 结果已经很好**（cosine 达标，无累积误差）→ 跳过，无需重新测
- **如果步骤 4 结果有累积误差**（疑似中间算子导致）→ 用 dump 工具（§A）重新跑一遍，
  拿到清理后 bmodel 的最终输出，与 HF 标准输出对比余弦相似度，确认精度正常

#### 1d. 再清理 demo 的其他代码

Converter 和 bmodel 确认 OK 后，再清理 LLM-TPU demo 的调试代码：

**chat.cpp**：
- 删除 `#include "utils.h"`（步骤 4 加的）
- 删除所有 `dump_net_to_file()` 调用
- 删除所有 debug `printf` / `fprintf` 语句
- 删除所有 `exit(0)` 提前终止

**pipeline.py**：
- 删除临时 dump 代码（如 `np.savez("debug_xxx.npz", ...)`）
- 删除临时 print 语句
- **保留** CLI 参数和正常的日志输出

**dump 工具（从 support/debug 拷贝的）**：
- 如果步骤 4 从 `LLM-TPU/support/debug/` 拷贝了 dump 工具到 `python_demo/` 目录
  （如 `utils.h`、`cnpy.cpp`、`cnpy.h` 等），确认不再需要后删除
- CMakeLists.txt 中如果为了链接这些工具做了修改（如添加 `cnpy.cpp`、链接 `z`），同步还原

#### 1e. 运行 demo 检查 warning

debug 代码清理完、干净 bmodel 重新编译后，实际跑一遍 demo（交互 + 单次推理），留意运行时打出的 **warning**：Python/transformers 的 deprecation、bmodel runtime 的 shape / 数值告警、未使用参数提示等。开发测试期可以忽略，但**发布前要逐一解决或确认无害**——warning 常是潜在问题的信号。发现的 warning 记到 `<model>_plan.md`。

### 2. 编写 README.md

基于 `<model>_plan.md` 中步骤 4 多设备精度验证已跑通的编译/运行命令，参考已有模型 README 格式写成，包含以下章节：

```markdown
# <模型名>

## 概述
- 模型介绍（参数量、架构、支持的模态）
- 支持的芯片和功能

## 模型架构
- 文本模型架构
- 视觉编码器架构（VLM）
- 特殊机制说明

## 下载预编译 bmodel
（如有预编译 bmodel，在此填写下载方式；无则留空）

## 编译 bmodel
1. 下载模型权重
2. Docker 环境
3. TPU-MLIR 编译
4. llm_convert.py 命令和参数说明

## 运行推理（Python）
1. 环境准备（pip 包版本）
2. 编译 C++ 库
3. 运行命令（交互模式 + 单次推理模式）

## CLI 参数表
| 参数 | 默认值 | 说明 |

## 常见问题
- Token 计算方式
- 支持的分辨率/帧数
- 内存需求
```

**「编译 bmodel」一节**：把 `<model>_plan.md` 中记录的编译命令（各目标芯片）填入，但**去掉 `--debug`**——`--debug` 是开发期保留 npz 权重给步骤 4 验证用的，发布的命令不需要。

### 3. 更新根目录 README

**文件**：`LLM-TPU/README.md`

三处修改：

1. **最新动态表**：添加一行
   ```
   | 🔥 **<日期>** | **<模型名>** 已支持 BM1684X / BM1688 → [查看](./models/<Model>/) |
   ```

2. **模型支持表**：在多模态或 LLM 表中添加条目

3. **完整目录索引**：在 models/ 列表中添加链接

### 4. 收尾检查

在收尾前，回溯检查所有需求是否已满足：

- [ ] **对照模板**：重新读取 `<model>_memory.md`，逐项确认部署需求都已实现
  - 支持的模态全部跑通？
  - 目标芯片都编译并测试了？
  - 编译模式（静态/动态）正确？
  - 历史记录支持（如需要）已实现？
- [ ] **对照适配计划**：读取 `<model>_plan.md`，检查：
  - 模型特异性的处理方案是否都已落地？
  - 待优化（workaround）章节是否有未记录的遗留？
  - 调试记录中是否有未解决的问题？
- [ ] **确认无遗漏**：
  - debug 代码是否全部清理？
  - bmodel 是否重新编译（不带中间算子）？
  - dump 工具文件是否已删除？
  - CMakeLists.txt 是否还原（如 debug 阶段改过）？
  - demo 实跑无未解决的 warning（开发期可忽略的，发布前已解决或确认无害）？

⚠️ **确认无误后才能收尾。** 如有遗留问题，先解决或在 <model>_plan.md 中标注为已知限制。

### 5. 清理 memory

从当前会话的项目记忆文件（`~/.claude/projects/` 下对应目录的 `memory/MEMORY.md`）中删除 llm-porting skill 注册的那一行条目。移植完成，不再需要自动恢复上下文。

## 完成标准

- [ ] debug 代码按顺序清理（Converter → 重新编译 bmodel → 验证 → demo 代码）
- [ ] bmodel 重新编译且不带中间算子
- [ ] README.md 已编写
- [ ] 根目录 README.md 已更新
- [ ] `<model>_plan.md` 调试记录章节已完整（汇总步骤 2-4 追加的内容），可作为复盘文档归档
- [ ] MEMORY.md 中 llm-porting 条目已删除
- [ ] 用户确认适配完成
