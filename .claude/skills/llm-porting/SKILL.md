---
name: llm-porting
description: "Use when: user wants to port a new LLM or VLM model for deployment on SOPHGO TPU (BM1684X/BM1688). Invoke with the model name; the skill copies the blank template (llm-porting-0-template.md) to <model>_memory.md for the user to fill."
---

# LLM/VLM 模型移植

> 一次启动，逐步推进。每步完成后等用户说「继续」再进入下一步。

## 前置条件

1. 模型权重已下载到本地（HuggingFace 格式）
2. 用户已确定要移植的模型名（启动时给出）
3. LLM-TPU 仓库（启动时自动检测，见下方）

> 主模板 `llm-porting-0-template.md` 永远保持空白，由 SKILL 复制使用，无需预填。

**per-model 文件落点**：记 `<work_dir>` = `<repo_root>/models/<model>/tmp/`（`<repo_root>` 为 LLM-TPU 仓库根目录）。`<model>_memory.md` 和 `<model>_plan.md` 都放此目录下，SKILL 在步骤 0 创建该目录。✓ `tmp/` 已被仓库 `.gitignore` 的 `tmp*/` 规则忽略，不会误提交（含环境凭据也安全）。

## 启动方式

```
按照 llm-porting skill，移植 [模型名]。
```

SKILL 会 `mkdir -p <work_dir>` 并把空白主模板 `llm-porting-0-template.md` 复制为 `<work_dir>/<model>_memory.md`，交给用户在副本上填写。

## 启动后立刻执行

**收到启动指令后，不要自行探索，严格按以下顺序执行：**

1. **确认仓库路径**：从本 SKILL.md 所在目录向上查找 `.git/` 定位仓库根目录，检查根目录下是否有 `models/` 目录且 `README.md` 含 "LLM-TPU" 关键字。命中 → 该根目录即为 `<repo_root>`，告知用户已自动确认。未命中 → 询问用户 LLM-TPU 仓库的本地路径
2. **复制模板**：`mkdir -p <work_dir>`，把 `llm-porting-0-template.md` 复制为 `<work_dir>/<model>_memory.md`（`<model>` 取自启动指令；预填"模型名称"和"LLM-TPU 路径"字段）。把副本交给用户填写，**停下等用户说「填好了」**
3. **读取 memory**：用户填好后读取 `<work_dir>/<model>_memory.md`，提取模型名、模态、环境信息等
4. **注册 MEMORY.md**：在当前会话的项目记忆文件（`~/.claude/projects/` 下对应目录的 `memory/MEMORY.md`）中追加一行：`- [llm-porting] <work_dir>/<model>_memory.md — 当前正在执行 llm-porting skill，每轮对话开始先 Read 此文件恢复上下文（完成后删除本条目）`
5. **进入步骤 1**：立刻读取 `llm-porting-1-analyze.md`，按其中的指令执行架构分析
6. **完成后停下**：展示 `<work_dir>/<model>_plan.md` 的内容，等用户确认
7. **等待指令**：用户说「继续」→ 读取 `llm-porting-2-converter.md` 并执行
8. **依次类推**：每个子 skill 完成后停下，等用户说「继续」再加载下一个

```
启动 → 自动检测 <repo_root>（未命中则问用户） → mkdir <work_dir> + 复制模板为 <model>_memory.md（预填 LLM-TPU 路径） → 用户填写 → 说「填好了」
     → 注册 MEMORY.md → 读 llm-porting-1-analyze.md → 执行 → 更新 memory → 停下等确认
用户说「继续」→ 读 llm-porting-2-converter.md → 执行 → 更新 memory → 停下等确认
用户说「继续」→ 读 llm-porting-3-pipeline.md → 执行 → 更新 memory → 停下等确认
用户说「继续」→ 读 llm-porting-4-verify.md → 执行 → 更新 memory → 停下等确认
用户说「继续」→ 读 llm-porting-5-cleanup.md → 执行（含清理 MEMORY.md 条目） → 完成
```

⚠️ **不要自己去找文件、探索目录、或做计划之外的事情。** 每步该做什么，子 skill 文件里写得很清楚，按指令执行即可。

## 步骤总览

| 步骤 | 子 Skill 文件 | 做什么 | 核心产物 |
|------|-------------|--------|---------|
| 1 | `llm-porting-1-analyze.md` | 读 HF 源码 → 架构分析 → 算子映射 → 识别模型特异性 | `<work_dir>/<model>_plan.md` |
| 2 | `llm-porting-2-converter.md` | 实现 Converter → 编译 bmodel | `<Model>Converter.py` + bmodel |
| 3 | `llm-porting-3-pipeline.md` | 实现 C++/Python 推理 pipeline → 跑通端到端 | `chat.cpp` + `pipeline.py` |
| 4 | `llm-porting-4-verify.md` | 精度验证（含多设备：同芯片查 pipeline、换芯片查 bmodel 内部） | 精度对比报告 |
| 5 | `llm-porting-5-cleanup.md` | 清 debug 代码 → 写 README → 收尾 | README + 清理完成 |

## 执行规则

- **串行推进**：每步完成后停下，等用户说「继续」再加载下一个子 skill
- **memory 维护**：`<model>_memory.md` 是贯穿全程的上下文记忆文件。**每步结束后**，更新其底部「移植进度备忘」章节（关键决策、关键命令、当前进度勾选）。**步骤 4 中每完成一个算子定位/修复**，也及时更新进度。这样 context 压缩后 AI 仍能从中恢复关键信息
- **per-model 文件**：每个模型在 `<work_dir>` 下产生两个文件——`<model>_memory.md`（用户填需求 + AI 维护进度）和 `<model>_plan.md`（AI 产出的完整调试记录）。主模板 `llm-porting-0-template.md` 永远空白
- **适配计划**：步骤 1 生成的 `<work_dir>/<model>_plan.md` 是活文档。**每个步骤执行过程中遇到的问题、解决方案、决策变更都要追加到该文档的调试记录章节。** 步骤 5 收尾时确认记录完整
- **模型特异性**：步骤 1 识别出的特异性内容是后续步骤的重点关注对象，需要人做决策
- **文本不可继承**：步骤 2/3/4 每个都按文本 → 视觉 → 语音的顺序执行（详见各子 Skill）
- **提交策略**：中间产物不逐步提交；Converter 跑通后可提交一次 tpu-mlir，Pipeline 跑通后可提交一次 LLM-TPU

## 三个环境

`<work_dir>/<model>_memory.md` 中需要填写三个环境，在适配过程中各步骤使用不同环境，**切勿混淆**：

| 环境 | 用途 | 使用步骤 |
|------|------|---------|
| **MLIR 编译环境** | 编译 bmodel（tpu-mlir 仓库） | 2, 4 |
| **TPU 运行环境** | 运行 pipeline（LLM-TPU 仓库） | 3, 4, 5 |
| **CUDA 环境** | 运行 HF 参考模型，获取标准输入/输出用于精度对比 | 1, 4 |

AI 在启动时应将三个环境信息记录到 `<work_dir>/<model>_plan.md` 中，后续步骤直接读取，不重复询问。
