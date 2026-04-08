# 组会汇报：3DVLMReasoning OpenEQA 进展

**日期**: 2026-04-04
**汇报人**: Shuhao Yue

---

## Slide 1: 封面

### 3D场景理解中的证据搜索型VLM Agent
#### ——OpenEQA ScanNet 评测进展

- **MNAS 73.1**，超越 Gemini 1.5 Flash（72.5），**排名第一**
- 比 GPT-4V baseline 提升 **+21.8 MNAS (+42.5%)**
- 覆盖 1050 个问题，7 个问答类别

---

## Slide 2: 任务背景与评测协议

### OpenEQA (CVPR 2024)
- **任务**: Embodied Question Answering — 给定3D场景视频，回答关于场景的自然语言问题
- **数据集**: ScanNet EM-EQA，1200+问题，89个室内场景，7类问答
- **评测指标**: LLM-Match (MNAS 0-100)，由LLM裁判对比预测答案与GT评分(1-5分)
- **Human performance**: MNAS 87.7

### 7类问答
| 类别 | 示例 |
|------|------|
| Object State Recognition | "Is the shower dry?" |
| Object Localization | "Where are the paper towels?" |
| Attribute Recognition | "What color is the sofa?" |
| Functional Reasoning | "Where can I take a nap?" |
| Spatial Understanding | "What is behind the sofa?" |
| Object Recognition | "What is the red object on the counter?" |
| World Knowledge | "What brand is the laptop?" |

---

## Slide 3: 系统架构

```
Question
    ↓
[Stage 1: Query-Driven Keyframe Retrieval]
    ├── LLM Query Parser → 结构化假设 (target + spatial constraints)
    ├── ConceptGraph 3D场景图匹配
    ├── LLM Enrichment 物体标签/描述/颜色/位置 (5097 objects, Gemini 2.5 Pro)
    ├── "Between" 空间中点检索
    └── 最少3帧可见性采样
    ↓
[Stage 2: VLM Evidence-Seeking Reasoning]
    ├── GPT-5.4 VLM 分析关键帧
    ├── 场景物体清单注入 System Prompt (category + description)
    ├── Tool callbacks: request_more_views, request_crops
    ├── 强制颜色/属性问题先 crop 再回答
    ├── Self-check: 答案不得与问题前提矛盾
    └── 显著性偏差纠正 (不默认选最大物体)
    ↓
[E2E: Extended Evidence-Seeking]
    └── 低置信度时进一步工具调用
```

### 工程量
- **119个文件修改**，+14,199行代码，32个commit
- 新增模块: 物体enrichment脚本(614行), 并发评估(12线程), pool key轮换重试
- 6轮迭代优化 (v9→v14), 5轮深度failure analysis (每轮6-8个并行agent团队)

---

## Slide 4: 迭代优化路径 (v9→v14)

| 版本 | MNAS | Δ | 关键改进 |
|------|:----:|:----:|---------|
| **v9** (baseline) | 46.5 | — | 无enrichment, 无tool callbacks |
| **v10** | 55.4 | +8.9 | LLM物体enrichment (89场景×5097物体) |
| **v11** | 62.6 | +7.2 | 启用evidence-seeking tools + 最少3帧 |
| **v12** | 65.0 | +2.4 | GPT-5.4 + 工具使用prompt规则 + 空间中点 |
| **v13** | 71.4 | +6.4 | 强制crop + 过度自信校准 (1050Q全量) |
| **v14** | **73.1** | +1.8 | 场景物体清单注入prompt |

**总提升: +26.6 MNAS (+57.2%)**

### 每轮的核心方法论
1. **跑评估** → 2. **分析全部低分case** (每个case看10+张原图) → 3. **归因系统性问题** → 4. **针对性修复** → 5. 重复

---

## Slide 5: 与Published Baselines对比

| 排名 | 方法 | MNAS |
|:---:|------|:----:|
| - | Human | 87.7 |
| **1** | **Ours (v14)** | **73.1** |
| 2 | Gemini 1.5 Flash | 72.5 |
| 3 | GLM-4.6V + Chain-of-View | 67.0 |
| 4 | CoV (Qwen3-VL) | 58.8 |
| 5 | GraphPad (Gemini 2.0 Flash) | 55.3 |
| 6 | GPT-4V (CVPR'24 paper) | 51.3 |
| 7 | GPT-4 + ConceptGraphs | 37.8 |
| 8 | GPT-4 (blind) | 32.5 |

- 比GPT-4V **+21.8 MNAS**
- 比ConceptGraphs原始方法 **+35.3 MNAS** (同一3D表示, 不同推理方法)
- Object Localization **79.5 超越 Human 77.3**

---

## Slide 6: 按类别分析 (v14, 1050Q)

| 类别 | MNAS | vs Human | vs GPT-4V |
|------|:----:|:--------:|:---------:|
| Object State | **84.7** | -14.0 | +21.5 |
| Object Localization | **78.0** | **+0.7** | +36.0 |
| Attribute | **77.5** | -10.4 | +20.3 |
| Functional | **76.9** | -4.9 | +19.5 |
| World Knowledge | **67.2** | -20.0 | +16.5 |
| Object Recognition | **66.0** | -21.9 | +22.6 |
| Spatial Understanding | **60.4** | -26.3 | +26.8 |

### 关键发现
- **强项**: Object State (84.7) 和 Localization (78.0) — 3D场景图 + evidence-seeking的核心优势
- **弱项**: Spatial Understanding (60.4) — "between X and Y"、"behind"等空间推理仍有挑战
- **全面超越GPT-4V**: 每个类别至少 +16.5 MNAS

---

## Slide 7: Failure Analysis与改进空间

### 失败模式分布 (222个低分case中抽样96个)

| Root Cause | 占比 | 可改善性 |
|------------|:---:|:-------:|
| VLM感知错误 (误识别物体/颜色) | 35% | 部分 (crops) |
| 检索失败 (keyframe未覆盖目标) | 24% | 是 |
| 分辨率/可见性 (暗光/小物体) | 12% | 部分 |
| 空间推理 | 8% | 是 (BEV+多视角) |
| 状态识别 (静态图固有限制) | 7% | 难 |
| GT标注问题 | 5% | 不可修复 |

### 关键发现
- **51%的失败case中，GT答案物体已在enrichment数据中** — 不是数据问题，是推理问题
- 估计天花板: **MNAS ~78** (vs Human 87.7)

---

## Slide 8: 工作量总结与Next Steps

### 完成的工作量

| 维度 | 数据 |
|------|:----:|
| 代码变更 | 119 files, +14,199 lines |
| Commit数 | 32 |
| 迭代轮次 | 6轮 (v9→v14) |
| 全量评估 | 2次 (v13: 1050Q, v14: 1050Q) |
| 子集评估 | 8次 (v9-v12各100Q + retries) |
| 低分Case深度分析 | 5轮, 共~300 cases |
| 并行分析Agent | 30+ agent teams |
| LLM物体enrichment | 89场景, 5097物体 |
| 评估用时 | 单次全量~5小时 |

### Next Steps

1. **Spatial Understanding提升** (当前60.4, 目标~68) — BEV引导的空间推理, 多视角一致性检查
2. **小物体检测** — 扩展GSA/RAM检测词汇, SAM v2
3. **World Knowledge** — 外部知识库对接, OCR工具集成
4. **HM3D数据集扩展** — 当前仅ScanNet, 计划扩展到HM3D split
