# OpenEQA Evaluation Analysis (2026-03-24)

本文基于 30 scenes x 4 questions = 120 题的 OpenEQA 官方评测结果，对 Stage 1 和 Stage 2 的系统性问题做全面诊断。

---

## 1. 评测配置

- **数据**: 30 个本地 ScanNet scenes，每 scene 4 个官方 OpenEQA 问题
- **Pipeline**: Stage 1 (keyframe retrieval) → Stage 2 (VLM reasoning, gpt-5.2) → E2E (Stage 2 + callbacks)
- **评测协议**: 官方 OpenEQA LLM-match (1-5 scale, normalized to 0-100)
- **Judge 模型**: gemini-2.5-pro via AzureChatOpenAI (gemini pool)
- **新增功能**: LLM query rewrite (`--llm-rewrite`), confidence guard (`--confidence-guard 0.6`), 8 workers parallel

## 2. 总体结果

| Metric | Stage 2 | E2E |
|--------|---------|-----|
| **Official Score (0-100)** | **44.4** | **47.5** (69 题 partial) |
| Fair comparison (同 69 题) | 44.6 | 47.5 (+2.9) |
| Completed | 86/120 (72%) | 88/120 (73%) |
| Insufficient evidence | 34/120 (28%) | 32/120 (27%) |
| Score=1 (完全错误) | 57/120 (48%) | — |
| Score=5 (完全正确) | 38/120 (32%) | — |

**结论**: 近半数题目完全答错 (score=1)，仅三分之一题目答对 (score=5)。

---

## 3. Stage 1 问题诊断

### 3.1 场景图标签质量极差 (根因 #1)

场景图来自 ConceptGraph 的 RAM-based 开放词汇检测。30 个 scene 共产出 2639 个 object labels，556 个 unique labels。

**问题 1: 大量非物体标签**

556 个 unique labels 中包含大量动词、形容词、颜色词等垃圾：

| 标签 | 出现次数 | 问题 |
|------|---------|------|
| `other item` | 30 | 最常见"分类"，说明检测器对大量物体无法识别 |
| `sit` | 24 | 动词，不是物体 |
| `white`, `black` | 22, 19 | 颜色词，不是物体 |
| `connect`, `hang`, `open` | 各 10+ | 动词/形容词 |
| `lead to`, `tight` | 各数次 | 完全无意义 |

**问题 2: 关键目标物体缺失**

OpenEQA 高频目标物体在场景图中的覆盖情况：

| 目标物体 | 场景图中 | 影响 |
|----------|---------|------|
| fire extinguisher | MISSING | 问 "What red object below windows" 无法定位 |
| cellphone | MISSING | 问 "Where is my cellphone" 无法定位 |
| blinds | MISSING | 问 "What color are the blinds" 无法定位 |
| carpet / rug | MISSING | 问 "What color is the carpet" 无法定位 |
| trash bin | MISSING | 问 "What is kept underneath table" 无法定位 |
| headphones | MISSING | 问 "What color is the wire for headphones" 无法定位 |
| flower pot | MISSING | 问 "What color of my flower pot" 无法定位 |
| soda can | MISSING | 问 "What color is the soda can" 无法定位 |
| charger | MISSING | 问 "Which electronic item next to fan" 无法定位 |
| patio chair | MISSING | 问 "What color are the patio chairs" 无法定位 |
| towel | FOUND (11x) | 少数能找到的目标 |
| lamp | FOUND (29x) | 覆盖较好 |
| pillow | FOUND (24x) | 覆盖较好 |
| shoe | FOUND (13x) | 覆盖较好 |
| stapler | FOUND (4x) | 部分覆盖 |

### 3.2 Stage 1 grounding 失败率高

| Stage 1 状态 | 数量 | 占比 | 含义 |
|-------------|------|------|------|
| `direct_grounded` | 40 | 33% | 找到了目标物体 |
| `proxy_grounded` | 55 | 46% | 只找到了附近物体做代理 |
| `context_only` | 25 | 21% | 目标和锚点都没找到 |

**67% 的题 Stage 1 没有找到真正的目标物体。**

Stage 1 grounding 状态对最终得分的影响：

| Stage 1 状态 | 平均得分 (0-100) | score=5 | score=1 |
|-------------|-----------------|---------|---------|
| direct_grounded | 59.4 | 18/40 (45%) | 12/40 (30%) |
| proxy_grounded | 44.5 | 17/55 (31%) | 26/55 (47%) |
| context_only | 20.0 | 3/25 (12%) | 19/25 (76%) |

### 3.3 direct_grounded 仍有 30% score=1

即使 Stage 1 声称找到了目标，仍有 12/40 (30%) 最终得分为 1。原因：

1. **选错了实例**: 场景中可能有多个同类物体，选到的不是问题指代的那个
   - "Where is my cellphone?" — S1 找到 cellphone，但选的 keyframe 看到的是床上的手机，GT 是桌上的
2. **Keyframe 角度/遮挡问题**: visibility index 只保证物体"在画面中"，不保证物体清晰可辨
   - "What color of sofa?" — S1 找到 couch，但选的角度下颜色失真 (pred: gray, GT: green)
3. **微小物体不可见**: 物体在 keyframe 中像素面积太小
   - "What color of the cloth hangar?" — S1 只选了 1 帧，衣架在画面中太小

### 3.4 LLM parser 效率低

- 每条 parse 需要 **15-35 秒**远端推理
- 输出的 hypothesis 经常把 target category 标注为 `UNKNOW`（还拼错了）
- 大量 prompt token 花在生成结构化 JSON schema 上，但 QueryExecutor 最终只用了 category 和 spatial relation 两个字段
- 同一个 scene 的不同问题需要重复加载场景、重复生成 BEV、重复 parse，没有 scene-level 缓存

---

## 4. Stage 2 问题诊断

### 4.1 48% 的题得分为 1

120 题中 57 题 score=1，其中：
- **34 题是 `insufficient_evidence`** — 模型诚实地说"看不到"，评分系统等价于答错
- **23 题是 `completed` 但答案完全错误** — 模型自信地给出了错误答案

`insufficient_evidence` 是稳赔策略：当前实现中，模型说"我看不到"等同于提交空答案，guaranteed score=1。但如果强制猜测，至少有部分概率猜对。

### 4.2 置信度标定有缺陷

| 置信度范围 | 数量 | 平均得分 | score=5 | score=1 |
|-----------|------|---------|---------|---------|
| < 0.4 | 34 | 1.00 | 0 | **34 (100%)** |
| 0.4-0.6 | 8 | 1.50 | 0 | 6 |
| 0.6-0.8 | 59 | 3.47 | 26 | **16 (27%)** |
| >= 0.8 | 19 | 4.32 | 12 | 1 |

- 低置信度 (< 0.4) 100% score=1: 置信度阈值 0.4 以下全是 "insufficient_evidence"
- **0.6-0.8 区间有 27% score=1**: 模型说"比较确定"但完全答错了——典型的 VLM 幻觉问题

### 4.3 E2E 回调循环价值有限

| 指标 | 数值 |
|------|------|
| 触发 tool call 的题 | 29/120 (24%) |
| 使用了 tool 的 E2E 得分 | 32.7 |
| 没用 tool 的 E2E 得分 | **50.9** |
| 从 insufficient_evidence 恢复 (score>=3) | 1/18 (6%) |

**使用 tool 的题得分反而更低。** 原因分析：

1. 触发 tool call 的前提是模型认为证据不足，这些题本身就更难
2. `request_more_views` 回调从同一个有缺陷的 visibility index 里再选几帧，选出来的帧大概率还是看不到目标物体（因为目标不在场景图里）
3. 额外帧引入更多噪声，可能干扰模型判断

### 4.4 模型非确定性导致 E2E 降级

Stage 2 和 E2E 是两次独立推理。当 E2E 没有触发任何 tool call 时（0 tool calls），它看到的输入与 Stage 2 完全相同，但由于模型采样的随机性，可能输出不同（更差）的答案。

已实现 `--confidence-guard 0.6` 缓解此问题。在 5-scene 测试中验证：
- Guard 触发 6/19 题，E2E 得分从 40.8 → 51.3 (+10.5)

### 4.5 Category 维度分析

| Category | Norm Score | N | score=1 | score=5 |
|----------|-----------|---|---------|---------|
| attribute recognition | 49.1 | 55 | 24 (44%) | 20 (36%) |
| object recognition | 43.1 | 54 | 26 (48%) | 16 (30%) |
| spatial understanding | 27.3 | 11 | 7 (64%) | 2 (18%) |

- **spatial understanding 最差** (27.3): 空间关系推理需要多视角理解物体相对位置，单帧 VQA 能力不足
- **object recognition 和 attribute recognition 接近**: 都受限于 keyframe 是否拍到了目标物体

---

## 5. 根因链条

```
ConceptGraph RAM detector 标签质量差
    ↓
556 个 labels 中大量垃圾 + 关键目标缺失
    ↓
Stage 1 QueryExecutor 找不到目标物体 (67% 非 direct)
    ↓
Keyframe 选择质量差 (proxy/context 选到的帧没有答案)
    ↓
Stage 2 看到的图片里没有需要回答的信息
    ↓
要么诚实说 "看不到" (insufficient_evidence → guaranteed score=1)
要么幻觉编答案 (completed 但 score=1)
```

**最大的瓶颈不在 Stage 2 的推理能力，而在 Stage 1 喂给它的视觉证据质量。**

---

## 6. 改进方向 (优先级排序)

### P0: 场景图标签质量

- 用更强的检测器（Grounding-DINO, Grounding-SAM 2）替代 RAM
- 或直接用 VLM (GPT-4o / Gemini) 做每帧的 open-vocabulary object captioning
- 清洗垃圾标签：过滤动词、形容词、颜色词

### P0: 建立 bypass Stage 1 的 baseline

- 直接均匀采样帧（每 N 帧取一帧）+ Stage 2 VQA，看 "天花板" 在哪
- 用 CLIP text-image 相似度直接选帧，绕过场景图
- 这能隔离 "Stage 1 有缺陷" vs "Stage 2 有缺陷" 的影响

### P1: 消灭 insufficient_evidence 的稳赔策略

- 当模型说 "insufficient_evidence" 时，强制让它给出最佳猜测 (best guess)
- 即使猜测不准确，也可能拿到 score=2-3，好过 guaranteed score=1
- 可以在 system prompt 中加 "You MUST always provide an answer, even if uncertain"

### P1: visibility index → 图片质量 index

- 当前 visibility index 只回答 "物体 X 在哪些帧中可见"
- 需要升级为 "物体 X 在哪些帧中**清晰可辨**"：考虑像素面积、遮挡比例、距离

### P2: Scene-level 缓存

- 同一 scene 的多个问题共享：BEV 图、场景加载、visibility index
- 当前每个问题都重新加载 scene (~2s)，30 scenes x 4 questions = 120 次重复加载

### P2: 减少 E2E 非确定性

- Stage 2 使用 temperature=0 减少采样随机性
- 或将 confidence guard 集成到 runtime 层而不是 post-hoc

### P3: spatial understanding 增强

- 当前 Stage 2 只看单/少量帧，缺乏 3D 空间感
- 可以将 BEV 俯视图、深度图作为额外输入
- 或引入 multi-view reasoning：将多帧按空间顺序排列，帮助模型建立空间关系

---

## 7. 参考数据

### 评测运行命令

```bash
# 30 scenes x 4 questions, 8 workers, LLM rewrite, confidence guard
.venv/bin/python -m agents.examples.openeqa_official_question_pilot \
  --num-scenes 30 --questions-per-scene 4 \
  --require-stage1-success \
  --workers 8 \
  --llm-rewrite \
  --confidence-guard 0.6 \
  --evaluate --eval-model gemini-2.5-pro \
  --output-root tmp/openeqa_eval_30x4
```

### 产出文件

- `tmp/openeqa_eval_30x4/official_batch_summary.json` — 未完成 (E2E eval 在 69/120 被限流)
- `tmp/openeqa_eval_30x4/official_predictions_stage2-metrics.json` — Stage 2 全量 120 题评分
- `tmp/openeqa_eval_30x4/official_predictions_e2e-metrics.json` — E2E 部分 69 题评分
- `tmp/openeqa_eval_30x4/runs/` — 每题的 stage1/stage2/e2e JSON artifacts
